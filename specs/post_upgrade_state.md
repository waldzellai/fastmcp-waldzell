# FastMCP TypeScript Port — Post-Upgrade State

This document describes the state of the codebase after implementing a comprehensive upgrade to achieve feature parity with the original Python FastMCP v2 where applicable, and to strengthen TypeScript ergonomics and production readiness.

## Overview

- Name: `fastmcp` (TypeScript)
- Runtime: Node.js 18+/22 ESM
- Package: MIT license
- Build: `tsup` with type declarations, ESM output
- Transports: stdio (default), Streamable HTTP, SSE
- CORS: Enabled by default for remote transports
- Schema: Standard Schema (Zod/ArkType/Valibot supported)
- Test: Vitest; in-memory transport tests included
- CLI: Dev, Inspect, and Generate features

## Major Areas

### 1) Tools

- API: `server.addTool({ name, description, parameters?, annotations?, timeoutMs?, execute })`
- Parameters: Accept Standard Schema v1; adapters for Zod, ArkType, Valibot
- Annotations: `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`, `title`
- Return types: string | ContentResult | TextContent | ImageContent | AudioContent
- Error surface: throw `UserError` for user-facing errors; other errors are captured as `isError: true`
- Timeout: Per-tool `timeoutMs` supported
- Logging & progress: `execute(_, { log, reportProgress, session, ctx })`
  - `log`: debug/info/warn/error (MCP notifications)
  - `reportProgress`: progress notifications with token propagation
  - `session`: typed auth/session payload (if authenticated)
  - `ctx`: extended server context (see Context)

### 2) Resources & Templates

- `server.addResource({ uri, name, mimeType?, load, description?, complete? })`
- `server.addResourceTemplate({ uriTemplate, name, mimeType?, arguments, load, description?, complete? })`
- Multiple contents: `load` can return one or many contents
- Binary/text: `blob` (base64) or `text`
- Argument completion: per-argument `complete(value)`; enum fuzzy-complete via Fuse

### 3) Prompts

- `server.addPrompt({ name, description?, arguments?, load, complete? })`
- Returns message content as user role
- Argument completion: per-argument `complete(value)` and enum fuzzy-complete

### 4) Transports

- `stdio`: default; no auth
- `sse`: `server.start({ transportType: 'sse', sse: { port, endpoint } })`
- `httpStream`: `server.start({ transportType: 'httpStream', httpStream: { port, endpoint } })`
- Ping policy: enabled by default for SSE/HTTP, disabled for stdio; configurable via `server` options
- Proxy: underlying remote transports implemented via `mcp-proxy`

### 5) Authentication & Sessions

- Servers can define `authenticate(request)` for SSE and HTTP Stream
- `authenticate` returns a typed payload attached to the session context
- Tools receive `session` in context; also accessible from `FastMCPSession`
- Client auth helpers (tokens/headers) available in generated manifests and docs

### 6) Context API (Server-side)

- `ctx.log`: structured logging to MCP client
- `ctx.reportProgress`: progress notifications
- `ctx.requestSampling(params)`: server-initiated LLM sampling requests (mirrors Python `ctx.sample`)
- `ctx.httpRequest(init)`: simple fetch-like helper with safe defaults (timeout, headers)
- `ctx.readResource(uri)`: read resources from own server
- All context helpers are available inside `execute`/`load` handlers via `ctx`

### 7) Roots Capability

- Opt-in by default; can be disabled per server with `roots: { enabled: false }`
- Emits `rootsChanged` on session when client updates roots
- Graceful fallback when client lacks roots support

### 8) Media Helpers

- `imageContent({ url | path | buffer })`: detects MIME, returns MCP image content
- `audioContent({ url | path | buffer })`: detects MIME, returns MCP audio content
- Helpers validate inputs and throw typed errors with helpful messages

### 9) Client Support & Testing

- Added a minimal Node client wrapper for developer ergonomics, tested against this server
- In-memory transport tests (no process management) for tools/resources/prompts
- Examples include stdio and remote transports; Inspector/CLI guidance provided

### 10) Composition & Proxying

- New `compose` utilities:
  - `mount(server: FastMCP, child: FastMCP, options?)` for live composition
  - `importServer(spec: { transport, url|command })` to statically copy tools/resources/prompts
- Proxy usage documented; composition tests included

### 11) OpenAPI/FastAPI Generation (Parity Feature)

- `FastMCP.fromOpenAPI(spec, options)`: generate tools/resources from OpenAPI
- `FastMCP.fromFastify(app, options)`: TypeScript-friendly FastAPI analogue
- Generated artifacts include descriptions, schemas, and annotations

### 12) CLI Enhancements

- `fastmcp dev <entry>`: run server with TTY-friendly output and auto-restart
- `fastmcp inspect <entry>`: open MCP Inspector against your server
- `fastmcp gen openapi <spec>`: generate a server scaffold from OpenAPI
- `fastmcp manifest <entry>`: emit MCP manifest including auth headers/transport

### 13) Configuration & DX

- Configurable ping intervals and levels; sane defaults per transport
- CORS enabled by default for SSE/HTTP stream
- Strong TypeScript types across public API; no `any` in public surface
- ESLint/Prettier/TypeScript strict settings enabled

### 14) Error Handling & Observability

- `UserError` for user-facing errors; `UnexpectedStateError` for internal invariant violations
- All public APIs validate inputs; MCP errors include `ErrorCode` and context
- Centralized logging; log level adjustable at runtime (`setLevel`)
- Server/session event emitters are strictly typed

### 15) Examples & Docs

- Expanded examples: tools with all three schema libs; media; progress; sampling; composition; openapi-generated server
- Readme sections updated with parity features and quickstarts
- Specs and guides for common deployments (stdio, Docker, serverless with stateless HTTP stream)

## Public API Summary

- `class FastMCP<TAuth>`
  - `addTool`, `addResource`, `addResourceTemplate`, `addPrompt`
  - `start({ transportType, ... })`, `stop()`
  - `sessions: FastMCPSession<TAuth>[]`
  - Events: `connect`, `disconnect`
- `class FastMCPSession<TAuth>`
  - `requestSampling(params)`
  - `clientCapabilities`, `loggingLevel`, `roots`, `server`
  - Events: `error`, `rootsChanged`
- Helpers: `imageContent`, `audioContent`
- Types: `Tool`, `Context`, `Resource`, `ResourceTemplate`, `Prompt`, `ServerOptions`, `LoggingLevel`, `ContentResult`
- Composition: `mount`, `importServer`
- Generators: `fromOpenAPI`, `fromFastify`

## Backwards Compatibility

- Existing `addTool`/`addResource`/`addPrompt` APIs remain stable
- New `ctx` helpers are additive; existing context usage remains valid
- Transport defaults unchanged; ping configuration is backward-compatible

## Limitations & Notes

- Client auth flows are helper-level vs. a full client library like Python v2; sufficient for typical deployments
- OpenAPI/Fastify generators cover common cases; advanced customization provided via hooks

## Example Snippet

```ts
server.addTool({
  name: "fetch-doc",
  description: "Fetches a document and summarizes with client LLM",
  parameters: z.object({ url: z.string().url() }),
  annotations: {
    readOnlyHint: true,
    openWorldHint: true,
    title: "Fetch & Summarize",
  },
  timeoutMs: 5000,
  async execute({ url }, { ctx, log, reportProgress }) {
    log.info("Fetching URL", { url });
    reportProgress({ progress: 10, total: 100 });

    const res = await ctx.httpRequest({ url, timeoutMs: 3500 });
    reportProgress({ progress: 50, total: 100 });

    const summary = await ctx.requestSampling({
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `Summarize: ${res.text.slice(0, 2000)}`,
          },
        },
      ],
      model: "auto",
    });

    reportProgress({ progress: 100, total: 100 });
    return summary.content;
  },
});
```
