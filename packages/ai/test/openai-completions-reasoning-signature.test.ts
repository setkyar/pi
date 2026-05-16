import { describe, expect, it } from "vitest";
import { convertMessages } from "../src/providers/openai-completions.js";
import type {
	AssistantMessage,
	Model,
	OpenAICompletionsCompat,
	Usage,
} from "../src/types.js";

const emptyUsage: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const defaultCompat = {
	supportsStore: true,
	supportsDeveloperRole: true,
	supportsReasoningEffort: true,
	supportsUsageInStreaming: true,
	maxTokensField: "max_completion_tokens",
	requiresToolResultName: false,
	requiresAssistantAfterToolResult: false,
	requiresThinkingAsText: false,
	requiresReasoningContentOnAssistantMessages: false,
	thinkingFormat: "openai",
	openRouterRouting: {},
	vercelGatewayRouting: {},
	zaiToolStream: false,
	supportsStrictMode: true,
	cacheControlFormat: undefined,
	sendSessionAffinityHeaders: false,
	supportsLongCacheRetention: true,
} satisfies Required<Omit<OpenAICompletionsCompat, "cacheControlFormat">> & {
	cacheControlFormat?: OpenAICompletionsCompat["cacheControlFormat"];
};

function buildModel(
	baseUrl = "http://127.0.0.1:1",
	compatOverride: Partial<OpenAICompletionsCompat> = {},
): Model<"openai-completions"> {
	return {
		id: "repro-model",
		name: "Repro Model",
		api: "openai-completions",
		provider: "repro-provider",
		baseUrl,
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 4096,
		compat: { ...defaultCompat, ...compatOverride },
	};
}

function buildAssistant(
	content: AssistantMessage["content"],
	model = buildModel(),
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: emptyUsage,
		stopReason: "stop",
		timestamp: 2,
	};
}

describe("openai-completions reasoning signature on replay", () => {
	it("should not set 'reasoning' field on assistant message when thinkingSignature is 'reasoning'", () => {
		// Kimi K2.6 streams reasoning via delta.reasoning, stored as thinkingSignature: "reasoning".
		// Using it as a message property (assistantMsg.reasoning = "text") causes 400 errors.
		const messages = convertMessages(
			buildModel(),
			{
				messages: [
					{ role: "user", content: "hello", timestamp: 1 },
					buildAssistant([
						{ type: "thinking", thinking: "I should say hello", thinkingSignature: "reasoning" },
						{ type: "text", text: "Hello there!" },
					]),
					{ role: "user", content: "what is 2+2?", timestamp: 3 },
				],
			},
			defaultCompat,
		);

		const assistantMsg = messages.find((m) => m.role === "assistant");
		expect(assistantMsg).toBeDefined();
		expect(assistantMsg).not.toHaveProperty("reasoning");
		expect(assistantMsg).toHaveProperty("content", "Hello there!");
	});

	it("should not set 'reasoning_text' field on assistant message when thinkingSignature is 'reasoning_text'", () => {
		const messages = convertMessages(
			buildModel(),
			{
				messages: [
					{ role: "user", content: "hello", timestamp: 1 },
					buildAssistant([
						{ type: "thinking", thinking: "I should say hello", thinkingSignature: "reasoning_text" },
						{ type: "text", text: "Hello there!" },
					]),
					{ role: "user", content: "what is 2+2?", timestamp: 3 },
				],
			},
			defaultCompat,
		);

		const assistantMsg = messages.find((m) => m.role === "assistant");
		expect(assistantMsg).not.toHaveProperty("reasoning_text");
	});

	it("should set 'reasoning_content' when thinkingSignature is 'reasoning_content' and requiresReasoningContentOnAssistantMessages is true", () => {
		// DeepSeek streams delta.reasoning_content and requires it on messages.
		const messages = convertMessages(
			buildModel(undefined, { requiresReasoningContentOnAssistantMessages: true }),
			{
				messages: [
					{ role: "user", content: "hello", timestamp: 1 },
					buildAssistant([
						{ type: "thinking", thinking: "I should say hello", thinkingSignature: "reasoning_content" },
						{ type: "text", text: "Hello there!" },
					]),
					{ role: "user", content: "what is 2+2?", timestamp: 3 },
				],
			},
			{ ...defaultCompat, requiresReasoningContentOnAssistantMessages: true },
		);

		const assistantMsg = messages.find((m) => m.role === "assistant");
		expect(assistantMsg).toHaveProperty("reasoning_content", "I should say hello");
	});

	it("should not set 'reasoning_content' when requiresReasoningContentOnAssistantMessages is false", () => {
		// Kimi and most providers don't require reasoning_content on messages.
		const messages = convertMessages(
			buildModel(),
			{
				messages: [
					{ role: "user", content: "hello", timestamp: 1 },
					buildAssistant([
						{ type: "thinking", thinking: "I should say hello", thinkingSignature: "reasoning_content" },
						{ type: "text", text: "Hello there!" },
					]),
					{ role: "user", content: "what is 2+2?", timestamp: 3 },
				],
			},
			defaultCompat,
		);

		const assistantMsg = messages.find((m) => m.role === "assistant");
		expect(assistantMsg).not.toHaveProperty("reasoning_content");
		expect(assistantMsg).not.toHaveProperty("reasoning");
	});

	it("should set reasoning_content fallback to empty string when no thinking blocks and requiresReasoningContentOnAssistantMessages is true", () => {
		// DeepSeek requires reasoning_content on all assistant messages when thinking is enabled.
		const messages = convertMessages(
			buildModel(undefined, { requiresReasoningContentOnAssistantMessages: true }),
			{
				messages: [
					{ role: "user", content: "hello", timestamp: 1 },
					buildAssistant([{ type: "text", text: "Hello there!" }]),
					{ role: "user", content: "what is 2+2?", timestamp: 3 },
				],
			},
			{ ...defaultCompat, requiresReasoningContentOnAssistantMessages: true },
		);

		const assistantMsg = messages.find((m) => m.role === "assistant");
		expect(assistantMsg).toHaveProperty("reasoning_content", "");
	});

	it("should set reasoning_content (not reasoning) when thinkingSignature is 'reasoning' and requiresReasoningContentOnAssistantMessages is true", () => {
		// When the delta field is 'reasoning' but the provider requires reasoning_content,
		// the thinking text should go to reasoning_content, not reasoning.
		const messages = convertMessages(
			buildModel(undefined, { requiresReasoningContentOnAssistantMessages: true }),
			{
				messages: [
					{ role: "user", content: "hello", timestamp: 1 },
					buildAssistant([
						{ type: "thinking", thinking: "I should say hello", thinkingSignature: "reasoning" },
						{ type: "text", text: "Hello there!" },
					]),
					{ role: "user", content: "what is 2+2?", timestamp: 3 },
				],
			},
			{ ...defaultCompat, requiresReasoningContentOnAssistantMessages: true },
		);

		const assistantMsg = messages.find((m) => m.role === "assistant");
		expect(assistantMsg).toHaveProperty("reasoning_content", "I should say hello");
		expect(assistantMsg).not.toHaveProperty("reasoning");
	});
});