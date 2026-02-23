// Adapter to match existing RAG return type
import { matchProtocol } from "../cloud-api/local/oasis-matcher";

export const getSystemPromptFromOasis = async (query: string): Promise<string> => {
    try {
        const result = await matchProtocol(query);
        // The service already returns the correct protocol text (or triage text)
        // We just return it to be injected as system prompt.
        // If it's a Triage, it returns the Triage prompt.
        
        // We might want to wrap it or just return as is.
        // The service returns: "Use epinephrine..." or "Ask ONE short question..."
        // But the service also constructs a full system prompt? 
        // No, the service returns 'text' which is the raw protocol text.
        // The Python script had a `build_system_prompt` function but the service implementation 
        // I wrote returns the raw text in 'text' field.
        
        // Let's wrap it here to match the user's prompt structure:
        const BASE_SYSTEM_PROMPT = `You are O.A.S.I.S., an emergency first-aid assistant.
Respond with short, direct commands only.
No markdown, no bullet points, no numbering, no symbols.
Speak in simple sentences. One instruction at a time.
Do not explain. Do not reassure. Just tell them what to do.

ACTIVE PROTOCOL:
${result.text}`;

        return BASE_SYSTEM_PROMPT;

    } catch (error) {
        console.error("OASIS Matcher failed:", error);
        return "";
    }
}
