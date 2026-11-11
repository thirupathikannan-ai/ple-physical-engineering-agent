
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Hybrid PLE engineering agent: planner, worker, independent verifier."""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback

import openai
import taskbridge


MODEL = os.environ.get("MODEL_NAME", "google-cloud/gemini-3.8-flash")
EFFORT = os.environ.get("REASONING_EFFORT", "high")
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "12000"))
MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "36"))
MAX_FIX_ROUNDS = 2
TOOL_RESULT_LIMIT = 18000

client = openai.AsyncOpenAI(
    base_url=os.environ.get("MODEL_BASE_URL"),
    api_key=os.environ.get("MODEL_API_KEY") or "none",
    max_retries=2,
    timeout=180,
)

BASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the engineering task container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_sec": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the task workspace.",
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
            "description": "Replace exact text in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]


def function_tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


MCP_TOOLS = [
    function_tool(
        "mcp_find",
        "Find available task-specific engineering tools by name or purpose.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    function_tool(
        "mcp_call",
        "Call a task-specific MCP tool using its exact listed name and arguments.",
        {
            "tool": {"type": "string"},
            "arguments": {"type": "object"},
        },
        ["tool", "arguments"],
    ),
]


def clip(value, limit=TOOL_RESULT_LIMIT):
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...[output truncated]...\n" + text[-limit // 2:]


class MCPServer(taskbridge.MCPClient):
    """Connect to one task MCP server."""

    def __init__(self, session, spec):
        super().__init__(session, spec["url"], client="ple-hybrid-agent")
        self.name = spec["name"]
        self.tools = []

    async def start(self):
        await self.initialize()
        self.tools = await self.list_tools()

    async def call(self, name, arguments):
        result = await self.call_tool(name, arguments)
        parts = []
        for item in result.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=False)[:5000])
        if not parts and result.get("structuredContent") is not None:
            parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
        output = "\n".join(parts) or "(no content)"
        return ("error: " if result.get("isError") else "") + clip(output)


class Agent:
    def __init__(self, session):
        self.session = session
        self.servers = []
        self.tool_map = {}
        self.catalog = []

    async def connect_mcp(self):
        for spec in getattr(self.session, "mcp_servers", []):
            try:
                server = MCPServer(self.session, spec)
                await server.start()
                self.servers.append(server)
                for item in server.tools:
                    name = item.get("name", "")
                    if not name:
                        continue
                    full_name = f"{server.name}__{name}"
                    self.tool_map[full_name] = (server, name, item)
                    self.catalog.append({
                        "server": server.name,
                        "name": name,
                        "description": item.get("description", ""),
                        "inputSchema": item.get("inputSchema", {}),
                    })
            except Exception as exc:
                print(f"[MCP] Could not connect to {spec.get('name', 'server')}: {exc}")

    def available_tools(self):
        # Keep the tool list small and discover task tools when needed.
        return BASE_TOOLS + MCP_TOOLS

    async def run_tool(self, name, args):
        try:
            if name == "bash":
                return clip(await self.session.exec(
                    args["command"],
                    timeout_sec=min(int(args.get("timeout_sec", 120)), 300),
                ))
            if name == "read_file":
                return clip(await self.session.read_file(args["path"]))
            if name == "write_file":
                await self.session.write_file(
                    args["path"], args["content"], append=bool(args.get("append", False))
                )
                return f"Wrote {args['path']}"
            if name == "edit_file":
                path = args["path"]
                old = args["old_text"]
                new = args["new_text"]
                content = await self.session.read_file(path)
                count = content.count(old)
                if count == 0:
                    return "error: old_text was not found"
                if count != 1 and not args.get("replace_all", False):
                    return f"error: old_text occurs {count} times; use replace_all=true"
                updated = content.replace(old, new) if args.get("replace_all") else content.replace(old, new, 1)
                await self.session.write_file(path, updated)
                return f"Updated {path}"
            if name == "mcp_find":
                query = args["query"].lower()
                matches = [
                    item for item in self.catalog
                    if query in item["name"].lower()
                    or query in item["description"].lower()
                    or query in item["server"].lower()
                ]
                return clip(json.dumps(matches[:30], ensure_ascii=False, indent=2))
            if name == "mcp_call":
                full_name = args["tool"]
                if full_name in self.tool_map:
                    server, real_name, _ = self.tool_map[full_name]
                else:
                    matches = [
                        (key, value) for key, value in self.tool_map.items()
                        if value[1] == full_name
                    ]
                    if len(matches) != 1:
                        return "error: tool name not found or ambiguous; call mcp_find first"
                    server, real_name, _ = matches[0][1]
                return await server.call(real_name, args.get("arguments", {}))
            return f"error: unknown tool {name}"
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    async def run_role(self, role, instruction, context="", max_calls=MAX_TOOL_CALLS):
        system = f"""
You are the {role} in a high-quality engineering agent for PLE.
Use the task container and task-specific MCP tools when appropriate.
The task workspace has no general internet access.
Read relevant files before making assumptions. Use calculations and tests
to support engineering conclusions. Respect units, constraints, safety,
and the requested output format. Create deliverables at the exact paths
requested by the task. Never claim a test passed unless you ran it.

Task-specific MCP tool catalog:
{json.dumps(self.catalog, ensure_ascii=False)[:24000]}

When finished, give a concise status and the paths of deliverables.
"""
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"TASK:\n{instruction}\n\n"
                    f"ROLE INSTRUCTIONS:\n{context or role}\n"
                ),
            },
        ]
        tools = self.available_tools()

        for _ in range(max_calls):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=MAX_OUTPUT_TOKENS,
                    reasoning_effort=EFFORT,
                )
                message = response.choices[0].message
            except Exception as exc:
                print(f"[{role}] model request failed: {exc}")
                return f"Model request failed: {exc}"

            assistant_message = {
                "role": "assistant",
                "content": message.content or "",
            }
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments or "{}",
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(assistant_message)

            if not message.tool_calls:
                return message.content or "No final response was returned."

            for call in message.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                    result = await self.run_tool(name, args)
                except Exception as exc:
                    result = f"error: invalid tool call: {exc}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": clip(result),
                })

        messages.append({
            "role": "user",
            "content": "Stop making tool calls now. Summarize progress, results, and remaining issues.",
        })
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_OUTPUT_TOKENS,
                reasoning_effort=EFFORT,
            )
            return response.choices[0].message.content or "Tool-call limit reached."
        except Exception as exc:
            return f"Tool-call limit reached; final summary failed: {exc}"


async def solve(session):
    started = time.monotonic()
    instruction = getattr(session, "instruction", "") or ""
    agent = Agent(session)

    print("[PLE] Connecting to task MCP servers...")
    await agent.connect_mcp()
    print(f"[PLE] Discovered {len(agent.catalog)} MCP tools.")

    plan = await agent.run_role(
        "Planner",
        instruction,
        context=(
            "Analyze the requirements and existing files. Produce a practical "
            "step-by-step plan, engineering method, assumptions, deliverable "
            "paths, and independent verification checks. Do not create or edit files."
        ),
        max_calls=8,
    )
    print("[PLE] Planning phase finished.")

    worker = await agent.run_role(
        "Worker",
        instruction,
        context=(
            "Implement the task fully. Follow the plan below, but correct it if "
            "the actual files or tests show a problem. Create the requested "
            "deliverables, run relevant checks, and reopen the final files.\n\n"
            f"PLAN:\n{plan}"
        ),
        max_calls=MAX_TOOL_CALLS,
    )
    print("[PLE] Worker phase finished.")

    verification = ""
    for round_number in range(MAX_FIX_ROUNDS + 1):
        verification = await agent.run_role(
            "Independent Verifier",
            instruction,
            context=(
                "Independently inspect the workspace and deliverables. Do not "
                "trust the worker's claims without checking files and tests. "
                "Check task requirements, numerical consistency, units, file "
                "formats, completeness, and reproducibility. Do not edit files. "
                "Return PASS only when evidence supports it; otherwise list "
                "specific defects and exact fixes.\n\n"
                f"Worker report:\n{worker}\n\n"
                f"Previous verification:\n{verification}"
            ),
            max_calls=10,
        )
        print(f"[PLE] Verification round {round_number + 1} finished.")

        if "PASS" in verification.upper() and "FAIL" not in verification.upper():
            break
        if round_number < MAX_FIX_ROUNDS:
            worker = await agent.run_role(
                "Worker",
                instruction,
                context=(
                    "Fix the concrete issues identified by the independent "
                    "verifier. Inspect files first, make the smallest correct "
                    "changes, rerun checks, and report evidence.\n\n"
                    f"Verifier findings:\n{verification}"
                ),
                max_calls=12,
            )

    elapsed = round(time.monotonic() - started, 1)
    print(f"[PLE] Finished in {elapsed}s.")
    return (
        f"Planner report:\n{plan}\n\n"
        f"Worker report:\n{worker}\n\n"
        f"Independent verification:\n{verification}\n\n"
        f"Elapsed seconds: {elapsed}"
    )


if __name__ == "__main__":
    taskbridge.run(solve)
