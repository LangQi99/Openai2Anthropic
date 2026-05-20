package gateway

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ============ Anthropic Request Types ============

type anthropicRequest struct {
	Model         string           `json:"model"`
	System        any              `json:"system,omitempty"`
	Messages      []anthropicInput `json:"messages"`
	Tools         []anthropicTool  `json:"tools,omitempty"`
	ToolChoice    any              `json:"tool_choice,omitempty"`
	MaxTokens     int              `json:"max_tokens,omitempty"`
	Temperature   *float64         `json:"temperature,omitempty"`
	Stream        bool             `json:"stream,omitempty"`
	StopSequences []string         `json:"stop_sequences,omitempty"`
}

type anthropicInput struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

// ============ OpenAI Responses API Types ============

type openAI2Request struct {
	Model           string          `json:"model"`
	Instructions    string          `json:"instructions,omitempty"`
	Input           []any           `json:"input"`
	Tools           []openAI2Tool   `json:"tools,omitempty"`
	ToolChoice      any             `json:"tool_choice,omitempty"`
	Stream          bool            `json:"stream,omitempty"`
	MaxOutputTokens int             `json:"max_output_tokens,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
}

type openAI2Tool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type openAI2Response struct {
	ID     string              `json:"id"`
	Object string              `json:"object"`
	Status string              `json:"status"`
	Output []openAI2OutputItem `json:"output"`
	Usage  struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error,omitempty"`
}

type openAI2OutputItem struct {
	Type    string `json:"type"`
	ID      string `json:"id,omitempty"`
	Role    string `json:"role,omitempty"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
	} `json:"content,omitempty"`
	Status    string `json:"status,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ============ Build Request ============

func buildOpenAI2Request(req anthropicRequest, model string) (openAI2Request, int, error) {
	estimatedInputTokens := 0

	result := openAI2Request{
		Model:  model,
		Stream: req.Stream,
		Input:  make([]any, 0),
	}

	// Convert system to instructions
	if systemText := extractSystemText(req.System); systemText != "" {
		result.Instructions = systemText
		estimatedInputTokens += estimateTextTokens(systemText)
	}

	// Convert messages to input
	for _, msg := range req.Messages {
		items, tokens := convertAnthropicMessageToOpenAI2Items(msg)
		result.Input = append(result.Input, items...)
		estimatedInputTokens += tokens
	}

	// Convert tools
	if len(req.Tools) > 0 {
		result.Tools = make([]openAI2Tool, 0, len(req.Tools))
		for _, tool := range req.Tools {
			result.Tools = append(result.Tools, openAI2Tool{
				Type:        "function",
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  sanitizeToolSchema(tool.InputSchema),
			})
		}

		// Map tool_choice
		result.ToolChoice = mapToolChoice(req.ToolChoice, hasToolResult(req.Messages))
	}

	if req.MaxTokens > 0 {
		result.MaxOutputTokens = req.MaxTokens
	}
	if req.Temperature != nil {
		result.Temperature = req.Temperature
	}

	return result, estimatedInputTokens, nil
}

func convertAnthropicMessageToOpenAI2Items(msg anthropicInput) ([]any, int) {
	var items []any
	var tokens int

	switch content := msg.Content.(type) {
	case string:
		textType := "input_text"
		if msg.Role == "assistant" {
			textType = "output_text"
		}
		items = append(items, map[string]any{
			"type": "message",
			"role": msg.Role,
			"content": []map[string]any{{
				"type": textType,
				"text": content,
			}},
		})
		tokens = estimateTextTokens(content)

	case []any:
		items, tokens = convertAnthropicBlocksToOpenAI2Items(content, msg.Role)
	}

	return items, tokens
}

func convertAnthropicBlocksToOpenAI2Items(blocks []any, role string) ([]any, int) {
	var items []any
	var messageParts []map[string]any
	var tokens int

	textType := "input_text"
	if role == "assistant" {
		textType = "output_text"
	}

	flushMessage := func() {
		if len(messageParts) == 0 {
			return
		}
		items = append(items, map[string]any{
			"type":    "message",
			"role":    role,
			"content": messageParts,
		})
		messageParts = nil
	}

	for _, block := range blocks {
		m, ok := block.(map[string]any)
		if !ok {
			continue
		}

		blockType, _ := m["type"].(string)
		switch blockType {
		case "text":
			text, _ := m["text"].(string)
			if text != "" {
				messageParts = append(messageParts, map[string]any{
					"type": textType,
					"text": text,
				})
				tokens += estimateTextTokens(text)
			}

		case "thinking":
			// Skip thinking blocks - Claude's internal reasoning

		case "tool_use":
			flushMessage()
			callID, _ := m["id"].(string)
			name, _ := m["name"].(string)
			args, _ := json.Marshal(m["input"])
			items = append(items, map[string]any{
				"type":      "function_call",
				"call_id":   callID,
				"name":      name,
				"arguments": string(args),
			})
			tokens += estimateTextTokens(string(args))

		case "tool_result":
			flushMessage()
			callID, _ := m["tool_use_id"].(string)
			output := toolResultToString(m["content"])
			items = append(items, map[string]any{
				"type":    "function_call_output",
				"call_id": callID,
				"output":  output,
			})
			tokens += estimateTextTokens(output)

		case "image":
			// Handle image blocks
			if part := convertImageBlockToOpenAI2(m); part != nil {
				messageParts = append(messageParts, part)
			}
		}
	}

	flushMessage()
	return items, tokens
}

func convertImageBlockToOpenAI2(block map[string]any) map[string]any {
	source, ok := block["source"].(map[string]any)
	if !ok {
		return nil
	}

	sourceType, _ := source["type"].(string)
	switch sourceType {
	case "base64":
		mediaType, _ := source["media_type"].(string)
		data, _ := source["data"].(string)
		if data == "" {
			return nil
		}
		if mediaType == "" {
			mediaType = "image/png"
		}
		return map[string]any{
			"type": "input_image",
			"image_url": map[string]any{
				"url": "data:" + mediaType + ";base64," + data,
			},
		}
	case "url":
		url, _ := source["url"].(string)
		if url == "" {
			return nil
		}
		return map[string]any{
			"type": "input_image",
			"image_url": map[string]any{
				"url": url,
			},
		}
	}
	return nil
}

func toolResultToString(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var parts []string
		for _, item := range v {
			if block, ok := item.(map[string]any); ok {
				if text, ok := block["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		data, _ := json.Marshal(v)
		return string(data)
	}
}

func mapToolChoice(toolChoice any, hasToolResult bool) any {
	if toolChoice == nil {
		if hasToolResult {
			return "auto"
		}
		return "required"
	}

	switch tc := toolChoice.(type) {
	case map[string]any:
		choiceType, _ := tc["type"].(string)
		switch choiceType {
		case "tool":
			if name, ok := tc["name"].(string); ok && name != "" {
				return map[string]any{
					"type": "function",
					"name": name,
				}
			}
		case "any":
			return "required"
		case "auto":
			return "auto"
		case "none":
			return "none"
		}
	case string:
		switch tc {
		case "any":
			return "required"
		default:
			return tc
		}
	}

	return "auto"
}

func hasToolResult(messages []anthropicInput) bool {
	for _, msg := range messages {
		blocks, ok := msg.Content.([]any)
		if !ok {
			continue
		}
		for _, block := range blocks {
			m, ok := block.(map[string]any)
			if !ok {
				continue
			}
			if t, _ := m["type"].(string); t == "tool_result" {
				return true
			}
		}
	}
	return false
}

// ============ Transform Response ============

func transformOpenAI2Response(raw []byte, requestedModel string) (map[string]any, error) {
	var resp openAI2Response
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, err
	}

	// Check for error in response
	if resp.Error != nil && resp.Error.Message != "" {
		return nil, fmt.Errorf("%s", resp.Error.Message)
	}

	var content []map[string]any
	stopReason := "end_turn"

	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" {
					content = append(content, splitThinkTaggedText(part.Text)...)
				}
			}
		case "function_call":
			var args map[string]any
			if item.Arguments != "" {
				_ = json.Unmarshal([]byte(item.Arguments), &args)
			}
			toolID := item.CallID
			if toolID == "" {
				toolID = item.ID
			}
			content = append(content, map[string]any{
				"type":  "tool_use",
				"id":    toolID,
				"name":  item.Name,
				"input": args,
			})
			stopReason = "tool_use"
		}
	}

	model := requestedModel
	if model == "" {
		model = resp.ID
	}

	return map[string]any{
		"id":            resp.ID,
		"type":          "message",
		"role":          "assistant",
		"model":         model,
		"content":       content,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  resp.Usage.InputTokens,
			"output_tokens": resp.Usage.OutputTokens,
		},
	}, nil
}

// ============ Helper Functions ============

func extractSystemText(system any) string {
	switch s := system.(type) {
	case string:
		return strings.TrimSpace(s)
	case []any:
		var parts []string
		for _, block := range s {
			if m, ok := block.(map[string]any); ok {
				if text, ok := m["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.TrimSpace(strings.Join(parts, "\n"))
	}
	return ""
}

func splitThinkTaggedText(text string) []map[string]any {
	const thinkTagOpen = "🐁"
	const thinkTagClose = " 笔者"

	var blocks []map[string]any
	for {
		openIdx := strings.Index(text, thinkTagOpen)
		if openIdx == -1 {
			if text != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": text})
			}
			return blocks
		}
		if openIdx > 0 {
			blocks = append(blocks, map[string]any{"type": "text", "text": text[:openIdx]})
		}
		text = text[openIdx+len(thinkTagOpen):]
		closeIdx := strings.Index(text, thinkTagClose)
		if closeIdx == -1 {
			if text != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": text})
			}
			return blocks
		}
		if closeIdx > 0 {
			blocks = append(blocks, map[string]any{"type": "thinking", "thinking": text[:closeIdx]})
		}
		text = text[closeIdx+len(thinkTagClose):]
	}
}

func resolveModel(requested string, available []string) string {
	requested = strings.TrimSpace(requested)
	if requested == "" {
		if len(available) > 0 {
			return available[0]
		}
		return "gpt-4.1-mini"
	}

	set := make(map[string]struct{}, len(available))
	for _, item := range available {
		set[item] = struct{}{}
	}
	if _, ok := set[requested]; ok {
		return requested
	}

	requestLower := strings.ToLower(requested)
	preferred := []string{}
	switch {
	case strings.Contains(requestLower, "haiku"):
		preferred = []string{"mini", "small", "haiku"}
	case strings.Contains(requestLower, "sonnet"):
		preferred = []string{"4.1", "4o", "sonnet", "turbo"}
	case strings.Contains(requestLower, "opus"):
		preferred = []string{"o1", "o3", "4.1", "4o"}
	}

	for _, keyword := range preferred {
		for _, item := range available {
			if strings.Contains(strings.ToLower(item), keyword) {
				return item
			}
		}
	}
	if len(available) > 0 {
		return available[0]
	}
	return requested
}

func sanitizeToolSchema(schema map[string]any) map[string]any {
	if schema == nil {
		return map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
	}
	schemaType, _ := schema["type"].(string)
	if schemaType == "object" {
		if _, ok := schema["properties"]; !ok {
			fixed := make(map[string]any, len(schema)+1)
			for k, v := range schema {
				fixed[k] = v
			}
			fixed["properties"] = map[string]any{}
			return fixed
		}
	}
	return schema
}

func estimateAnthropicInputTokens(req anthropicRequest) int {
	total := estimateTextTokens(extractSystemText(req.System))
	for _, message := range req.Messages {
		switch content := message.Content.(type) {
		case string:
			total += estimateTextTokens(content)
		case []any:
			for _, item := range content {
				block, ok := item.(map[string]any)
				if !ok {
					continue
				}
				switch block["type"] {
				case "text":
					if text, _ := block["text"].(string); text != "" {
						total += estimateTextTokens(text)
					}
				case "tool_result":
					total += estimateTextTokens(toolResultToString(block["content"]))
				case "tool_use":
					raw, _ := json.Marshal(block["input"])
					total += estimateTextTokens(string(raw))
				}
			}
		}
	}
	return total
}