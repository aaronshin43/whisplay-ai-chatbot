import { matchProtocolLocal, OasisMatchResult } from "./oasis-matcher-node";

// Simply redirect to the local Node.js implementation
export { OasisMatchResult };

export const matchProtocol = async (query: string): Promise<OasisMatchResult> => {
    return await matchProtocolLocal(query);
};
