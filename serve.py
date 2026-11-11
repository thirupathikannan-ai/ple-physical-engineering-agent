
import asyncio
import json
import os
import time
import traceback

import openai
import taskbridge


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = os.environ.get(
    "MODEL_NAME",
    "google-cloud/gemini-3.8-flash",
)

EFFORT = os.environ.get("REASONING_EFFORT", "high")

MAX_OUTPUT_TOKENS = 12000
MAX_TOOL_CALLS = 36
MAX_FIX_ROUNDS = 2
TOOL_RESULT_LIMIT = 18000

client = openai.AsyncOpenAI(
    base_url=os.environ.get("MODEL_BASE_URL"),
    api_key=os.environ.get("MODEL_API_KEY", "none"),
    max_retries=2,
    timeout=180,
)


# ============================================================
# TOOL DEFINITIONS
# ============================================================

BASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command inside the engineering task "
                "container. Use this for calculations, scripts, "
                "testing, and inspecting files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute.",
                    },
                    "timeout_sec": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the task container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact text occurrence inside an existing file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {
                        "type": "string",
                        "description": "Exact text to find.",
                    },
                    "new": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "old", "new"],
            },
        },
    },
]


MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mcp_find",
            "description": (
                "Search the available MCP tool catalog. "
                "Use this to discover tools for the current task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Tool name or capability to search for.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_call",
            "description": (
                "Call a discovered MCP tool using its server name, "
                "tool name, and JSON arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string"},
                    "tool": {"type": "string"},
                    "arguments": {
                        "type": "object",
                        "description": "Arguments required by the MCP tool.",
                    },
                },
                "required": ["server", "tool", "arguments"],
            },
        },
    },
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clip(value, limit=TOOL_RESULT_LIMIT):
    """Convert a tool result into limited text."""

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)

    if len(value) > limit:
        value = value[:limit] + "\n...[output truncated]"

    return value


def json_result(value):
    """Convert MCP results into readable JSON text."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


# ============================================================
# MCP SERVER WRAPPER
# ============================================================

class MCPServer(taskbridge.MCPClient):

    def __init__(self, session, spec):
        super().__init__(
            session,
            spec["url"],
            client="ple-hybrid-agent",
        )

        self.name = spec["name"]
        self.tools = []

    async def connect(self):
        await self.initialize()
        self.tools = await self.list_tools()
        return self.tools

    async def call(self, tool_name, arguments):
        return await self.call_tool(tool_name, arguments)


# ============================================================
# AGENT
# ============================================================

class Agent:

    def __init__(self, session):
        self.session = session
        self.servers = {}
        self.catalog = []
        self.tool_calls = 0
        self.start_time = time.monotonic()

    # --------------------------------------------------------
    # MCP DISCOVERY
    # --------------------------------------------------------

    async def connect_mcp(self):

        for spec in self.session.mcp_servers:

            name = spec.get("name")
            url = spec.get("url")

            if not name or not url:
                continue

            try:
                server = MCPServer(self.session, spec)

                tools = await server.connect()

                self.servers[name] = server

                for tool in tools:
                    self.catalog.append({
                        "server": name,
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "inputSchema": tool.get("inputSchema", {}),
                    })

                await self.session.log(
                    f"MCP connected: {name}, tools discovered: {len(tools)}"
                )

            except Exception as exc:
                await self.session.log(
                    f"MCP connection failed for {name}: "
                    f"{type(exc).__name__}: {exc}"
                )

    # --------------------------------------------------------
    # TOOL EXECUTION
    # --------------------------------------------------------

    async def run_tool(self, name, args):

        try:

            # --------------------------------------------
            # BASH
            # --------------------------------------------

            if name == "bash":

                result = await self.session.exec(
                    args["command"],
                    timeout_sec=min(
                        int(args.get("timeout_sec", 120)),
                        300,
                    ),
                )

                return clip(
                    f"Exit code: {result.get('exit_code')}\n"
                    f"Output:\n{result.get('output', '')}"
                )

            # --------------------------------------------
            # READ FILE
            # --------------------------------------------

            if name == "read_file":

                content = await self.session.read_file(args["path"])

                return clip(
                    content.decode("utf-8", errors="replace")
                )

            # --------------------------------------------
            # WRITE FILE
            # --------------------------------------------

            if name == "write_file":

                size = await self.session.write_file(
                    args["path"],
                    args["content"],
                    append=bool(args.get("append", False)),
                )

                return f"File written successfully. Size: {size} bytes."

            # --------------------------------------------
            # EDIT FILE
            # --------------------------------------------

            if name == "edit_file":

                path = args["path"]
                old = args["old"]
                new = args["new"]

                content = (
                    await self.session.read_file(path)
                ).decode(
                    "utf-8",
                    errors="replace",
                )

                count = content.count(old)

                if count == 0:
                    return (
                        f"Edit failed: exact text was not found in {path}."
                    )

                if count > 1:
                    return (
                        f"Edit failed: found {count} occurrences. "
                        "Provide a more specific old text."
                    )

                updated = content.replace(old, new, 1)

                size = await self.session.write_file(
                    path,
                    updated,
                )

                return (
                    f"File edited successfully: {path}. "
                    f"New size: {size} bytes."
                )

            # --------------------------------------------
            # MCP TOOL SEARCH
            # --------------------------------------------

            if name == "mcp_find":

                query = args.get("query", "").lower()

                matches = []

                for item in self.catalog:

                    searchable = (
                        item["server"] + " "
                        + item["name"] + " "
                        + item["description"]
                    ).lower()

                    if query in searchable:
                        matches.append(item)

                if not matches:
                    return "No matching MCP tools found."

                return clip(matches, 12000)

            # --------------------------------------------
            # MCP TOOL CALL
            # --------------------------------------------

            if name == "mcp_call":

                server_name = args["server"]
                tool_name = args["tool"]
                arguments = args.get("arguments", {})

                server = self.servers.get(server_name)

                if server is None:
                    return f"MCP server not available: {server_name}"

                allowed = any(
                    item["server"] == server_name
                    and item["name"] == tool_name
                    for item in self.catalog
                )

                if not allowed:
                    return (
                        f"MCP tool not found: {server_name}/{tool_name}"
                    )

                result = await server.call(
                    tool_name,
                    arguments,
                )

                return clip(json_result(result))

            return f"Unknown tool: {name}"

        except taskbridge.Cancelled:
            raise

        except Exception as exc:

            return (
                f"Tool error ({name}): "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # MCP CATALOG DESCRIPTION
    # --------------------------------------------------------

    def catalog_text(self):

        if not self.catalog:
            return "No MCP tools are available."

        lines = []

        for item in self.catalog:

            lines.append(
                f"- Server: {item['server']}\n"
                f"  Tool: {item['name']}\n"
                f"  Description: {item['description']}\n"
                f"  Input schema: "
                f"{json.dumps(item['inputSchema'], ensure_ascii=False)}"
            )

        return "\n".join(lines)

    # --------------------------------------------------------
    # ROLE EXECUTION
    # --------------------------------------------------------

    async def run_role(
        self,
        role,
        instruction,
        context="",
        max_calls=MAX_TOOL_CALLS,
    ):

        system_prompt = f"""
You are the {role} in a professional autonomous
physical-engineering problem-solving agent.

Your objective is to solve the user's engineering task
accurately, safely, and with verifiable results.

GENERAL RULES:

1. Read and understand the complete task instruction.
2. Use calculations and engineering reasoning.
3. Never invent measurements, test results, or evidence.
4. Use available tools when necessary.
5. Check units, assumptions, and boundary conditions.
6. Save required deliverables in the task workspace.
7. Verify files and calculations before finishing.
8. Respect the task deadline.
9. Report limitations honestly.

AVAILABLE MCP TOOLS:

{self.catalog_text()}

ROLE:

{role}

TASK INSTRUCTION:

{instruction}

ADDITIONAL CONTEXT:

{context}
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Begin your {role} work now. "
                    "Explain your actions through tool calls and "
                    "finish with a concise summary."
                ),
            },
        ]

        available_tools = BASE_TOOLS + MCP_TOOLS

        calls_used = 0
        last_text = ""

        while calls_used < max_calls:

            if self.session.cancelled.is_set():
                raise taskbridge.Cancelled()

            left = self.session.time_left()

            if left is not None and left <= 10:
                last_text += "\nTime limit is approaching."
                break

            try:

                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto",
                    max_tokens=MAX_OUTPUT_TOKENS,
                    reasoning_effort=EFFORT,
                )

            except Exception as exc:

                await self.session.log(
                    f"{role} model request failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                last_text += (
                    f"\nModel request error: {type(exc).__name__}: {exc}"
                )

                break

            if not response.choices:
                last_text += "\nModel returned no choices."
                break

            message = response.choices[0].message

            if message.content:
                last_text += str(message.content)

            if not message.tool_calls:

                if message.content:
                    break

                last_text += "\nModel returned an empty response."
                break

            messages.append(
                message.model_dump(exclude_none=True)
            )

            for tool_call in message.tool_calls:

                if calls_used >= max_calls:
                    break

                calls_used += 1
                self.tool_calls += 1

                tool_name = tool_call.function.name

                try:
                    arguments = json.loads(
                        tool_call.function.arguments or "{}"
                    )

                    if not isinstance(arguments, dict):
                        raise ValueError(
                            "Tool arguments must be a JSON object."
                        )

                except (ValueError, TypeError) as exc:

                    result = f"Invalid tool arguments: {exc}"

                else:

                    await self.session.log(
                        f"{role}: executing tool {tool_name}"
                    )

                    result = await self.run_tool(
                        tool_name,
                        arguments,
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": clip(result),
                })

        return {
            "role": role,
            "summary": last_text,
            "tool_calls": calls_used,
            "elapsed_seconds": round(
                time.monotonic() - self.start_time,
                2,
            ),
        }


# ============================================================
# HYBRID AGENT WORKFLOW
# ============================================================

async def solve(session):

    agent = Agent(session)

    await session.log(
        "PLE Hybrid Engineering Agent started."
    )

    await session.log(
        f"Selected model: {MODEL}"
    )

    # --------------------------------------------------------
    # STEP 1: CONNECT MCP SERVERS
    # --------------------------------------------------------

    await agent.connect_mcp()

    await session.log(
        f"Available MCP tools: {len(agent.catalog)}"
    )

    instruction = session.instruction

    # --------------------------------------------------------
    # STEP 2: PLANNER
    # --------------------------------------------------------

    await session.log(
        "Starting independent Planner."
    )

    planner = await agent.run_role(
        role="PLANNER",
        instruction=instruction,
        context=(
            "Create a detailed engineering solution plan. "
            "Identify deliverables, calculations, tools, "
            "assumptions, risks, and verification requirements. "
            "Do not modify final deliverables during planning."
        ),
        max_calls=8,
    )

    plan = planner["summary"]

    await session.log(
        "Planner completed."
    )

    # --------------------------------------------------------
    # STEP 3: WORKER
    # --------------------------------------------------------

    await session.log(
        "Starting Engineering Worker."
    )

    worker = await agent.run_role(
        role="WORKER",
        instruction=instruction,
        context=(
            "Follow the Planner's plan below.\n\n"
            f"PLANNER OUTPUT:\n{plan}\n\n"
            "Execute the engineering task. "
            "Perform necessary calculations, create the required "
            "files, and test the deliverables. "
            "Do not claim that a test passed unless you actually "
            "ran it and observed the result."
        ),
        max_calls=MAX_TOOL_CALLS,
    )

    worker_summary = worker["summary"]

    await session.log(
        "Worker completed."
    )

    # --------------------------------------------------------
    # STEP 4: INDEPENDENT VERIFIER
    # --------------------------------------------------------

    verification = None
    fix_summaries = []

    for round_number in range(MAX_FIX_ROUNDS + 1):

        left = session.time_left()

        if left is not None and left <= 20:
            await session.log(
                "Skipping additional verification/fix work "
                "because the deadline is close."
            )
            break

        await session.log(
            f"Starting Independent Verifier. "
            f"Round {round_number + 1}."
        )

        verification = await agent.run_role(
            role="INDEPENDENT VERIFIER",
            instruction=instruction,
            context=(
                "You are an independent engineering verifier. "
                "Do not blindly trust the Planner or Worker. "
                "Inspect the actual files and calculations using "
                "available tools.\n\n"
                f"PLANNER PLAN:\n{plan}\n\n"
                f"WORKER SUMMARY:\n{worker_summary}\n\n"
                "Check correctness, completeness, units, "
                "assumptions, deliverable paths, and test evidence. "
                "Clearly state PASS or FAIL. "
                "If something is wrong, list specific issues "
                "and the exact corrections required."
            ),
            max_calls=12,
        )

        verification_text = verification["summary"]

        await session.log(
            "Verifier completed."
        )

        # A conservative decision: do not assume success from
        # the presence of the word PASS alone.
        upper = verification_text.upper()

        has_pass = "PASS" in upper
        has_fail = "FAIL" in upper

        if has_pass and not has_fail:
            break

        if round_number >= MAX_FIX_ROUNDS:
            break

        left = session.time_left()

        if left is not None and left <= 30:
            break

        # ----------------------------------------------------
        # STEP 5: FIX LOOP
        # ----------------------------------------------------

        await session.log(
            f"Starting Worker correction round {round_number + 1}."
        )

        fix = await agent.run_role(
            role="WORKER - CORRECTION",
            instruction=instruction,
            context=(
                f"Original plan:\n{plan}\n\n"
                f"Previous worker summary:\n{worker_summary}\n\n"
                f"Verifier findings:\n{verification_text}\n\n"
                "Fix the specific problems identified by the "
                "verifier. Recheck the corrected files and "
                "calculations. Do not remove correct work."
            ),
            max_calls=MAX_TOOL_CALLS,
        )

        fix_summaries.append(fix["summary"])

        worker_summary = (
            worker_summary
            + "\n\nCORRECTION ROUND:\n"
            + fix["summary"]
        )

    # --------------------------------------------------------
    # STEP 6: FINAL REPORT
    # --------------------------------------------------------

    final_report = f"""
PLE HYBRID ENGINEERING AGENT - FINAL REPORT

MODEL:
{MODEL}

PLANNER SUMMARY:
{plan}

WORKER SUMMARY:
{worker_summary}

VERIFICATION SUMMARY:
{
    verification["summary"]
    if verification
    else "Verification was not completed."
}

CORRECTION ROUNDS:
{len(fix_summaries)}

TOTAL TOOL CALLS:
{agent.tool_calls}

ELAPSED TIME:
{round(time.monotonic() - agent.start_time, 2)} seconds

FINAL STATUS:
See the independent verifier's findings above.
Do not treat unverified deliverables as successful.
"""

    await session.log(
        "PLE Hybrid Engineering Agent finished."
    )

    session.final_message = final_report

    return final_report


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    taskbridge.run(solve)
