import { matchProtocol } from "../cloud-api/local/oasis-matcher";

export interface OasisResult {
    protocolId: string;
    protocolText: string;
    systemPrompt: string;
    isTriage: boolean;
    score: number;
}

const buildFollowUpSystemPrompt = (protocolText: string): string => {
    return `You are an emergency assistant. Answer the user's follow-up question using ONLY the protocol below. Quote relevant sentences from the protocol directly. Keep it short.

PROTOCOL:
${protocolText}`;
};

const buildTriageSystemPrompt = (): string => {
    return `You are an emergency assistant. The situation is unclear. Ask ONE short question to find out what is happening. For example: Is there bleeding? Can they breathe? Are they conscious? What happened?`;
};

export const matchOasisProtocol = async (query: string): Promise<OasisResult> => {
    try {
        const result = await matchProtocol(query);
        const isTriage = result.triage;
        return {
            protocolId: result.protocol_id,
            protocolText: result.text,
            systemPrompt: isTriage
                ? buildTriageSystemPrompt()
                : buildFollowUpSystemPrompt(result.text),
            isTriage,
            score: result.score,
        };
    } catch (error) {
        console.error("OASIS Matcher failed:", error);
        return {
            protocolId: "TRIAGE",
            protocolText: "",
            systemPrompt: buildTriageSystemPrompt(),
            isTriage: true,
            score: 0,
        };
    }
};

// Legacy compat
export const getSystemPromptFromOasis = async (query: string): Promise<string> => {
    const result = await matchOasisProtocol(query);
    return result.isTriage ? result.systemPrompt : result.protocolText;
};

export { buildFollowUpSystemPrompt };
