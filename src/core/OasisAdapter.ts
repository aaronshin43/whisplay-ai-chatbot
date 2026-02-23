import { matchProtocol } from "../cloud-api/local/oasis-matcher";

const buildOasisSystemPrompt = (protocolText: string): string => {
    return `You are O.A.S.I.S., an emergency first-aid assistant.
Respond with short, direct commands only.
No markdown, no bullet points, no numbering, no symbols.
Speak in simple sentences. One instruction at a time.
Do not explain. Do not reassure. Just tell them what to do.

ACTIVE PROTOCOL:
${protocolText}`;
};

export const getSystemPromptFromOasis = async (query: string): Promise<string> => {
    try {
        const result = await matchProtocol(query);
        return buildOasisSystemPrompt(result.text);
    } catch (error) {
        console.error("OASIS Matcher failed:", error);
        return "";
    }
};

export { buildOasisSystemPrompt };
