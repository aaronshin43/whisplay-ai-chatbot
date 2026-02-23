import axios from "axios";
import * as fs from "fs";
import * as path from "path";
import { isEmpty } from "lodash";
import {
  shouldResetChatHistory,
  systemPrompt,
  updateLastMessageTime,
} from "../../config/llm-config";
import { llmTools, llmFuncMap } from "../../config/llm-tools";
import dotenv from "dotenv";
import {
  Message,
  OllamaFunctionCall,
  OllamaMessage,
  ToolReturnTag,
} from "../../type";
import { ChatWithLLMStreamFunction, SummaryTextWithLLMFunction } from "../interface";
import { chatHistoryDir } from "../../utils/dir";
import moment from "moment";
import {
  extractToolResponse,
  stimulateStreamResponse,
} from "../../config/common";
import { defaultPortMap } from "./common";

dotenv.config();

// Ollama LLM configuration
const ollamaEndpoint =
  process.env.OLLAMA_ENDPOINT || `http://localhost:${defaultPortMap.ollama}`;
const ollamaModel = process.env.OLLAMA_MODEL || "deepseek-r1:1.5b";
const ollamaEnableTools = process.env.OLLAMA_ENABLE_TOOLS === "true";
const ollamaPredictNum = process.env.OLLAMA_PREDICT_NUM
  ? parseInt(process.env.OLLAMA_PREDICT_NUM)
  : undefined;
const enableThinking = process.env.ENABLE_THINKING === "true";

const llmServer = process.env.LLM_SERVER || "";

const chatHistoryFileName = `ollama_chat_history_${moment().format(
  "YYYY-MM-DD_HH-mm-ss",
)}.json`;

const messages: OllamaMessage[] = [
  {
    role: "system",
    content: systemPrompt,
  },
];

// Minimal prompt for keep-alive only (model load). Real requests use OASIS or config systemPrompt.
const keepAliveSystemPrompt = "You are a helpful assistant.";
const keepAliveOllama = () => {
  axios
    .post(`${ollamaEndpoint}/api/chat`, {
      model: ollamaModel,
      messages: [
        {
          role: "system",
          content: keepAliveSystemPrompt,
        },
      ],
      options: {
        temperature: 0.7,
        num_predict: 1,
      },
      think: false,
      stream: false,
      tools: ollamaEnableTools ? llmTools : undefined,
      keep_alive: -1,
    })
    .then((response) => {
      console.log("Ollama keep-alive response:", response.data);
    })
    .catch((err) => {
      console.error("Error initializing Ollama model:", err.message);
    });
};

if (llmServer.trim().toLowerCase() === "ollama") {
  // initialize request to ollama server with empty prompt, to load the model into memory
  keepAliveOllama();
}

const resetChatHistory = (): void => {
  messages.length = 0;
  messages.push({
    role: "system",
    content: systemPrompt,
  });
};

const chatWithLLMStream: ChatWithLLMStreamFunction = async (
  inputMessages: Message[] = [],
  partialCallback: (partialAnswer: string) => void,
  endCallback: () => void,
  partialThinkingCallback?: (partialThinking: string) => void,
  invokeFunctionCallback?: (functionName: string, result?: string) => void,
): Promise<void> => {
  const hasOwnSystem = inputMessages.length > 0 && inputMessages[0].role === "system";
  const sameSystemAsBefore = hasOwnSystem
    && messages.length > 0
    && messages[0].role === "system"
    && messages[0].content === inputMessages[0].content;

  if (hasOwnSystem) {
    if (sameSystemAsBefore && !shouldResetChatHistory()) {
      // Same system prompt, continuing conversation → KV cache friendly (only append user)
      const nonSystemMessages = inputMessages.filter(m => m.role !== "system");
      messages.push(...(nonSystemMessages as OllamaMessage[]));
    } else {
      // New system prompt or session reset → fresh start
      messages.length = 0;
      messages.push(...(inputMessages as OllamaMessage[]));
    }
  } else {
    if (shouldResetChatHistory()) {
      messages.length = 0;
      messages.push({ role: "system", content: systemPrompt } as OllamaMessage);
    }
    messages.push(...(inputMessages as OllamaMessage[]));
  }
  updateLastMessageTime();
  let endResolve: () => void = () => {};
  const promise = new Promise<void>((resolve) => {
    endResolve = resolve;
  }).finally(() => {
    // save chat history to file
    fs.writeFileSync(
      path.join(chatHistoryDir, chatHistoryFileName),
      JSON.stringify(messages, null, 2),
    );
  });
  let partialAnswer = "";
  let partialThinking = "";
  const functionCallsPackages: OllamaFunctionCall[][] = [];

  let buffer = "";

  try {
    const response = await axios.post(
      `${ollamaEndpoint}/api/chat`,
      {
        model: ollamaModel,
        messages: messages.map((msg) => ({
          role: msg.role,
          content: msg.content,
        })),
        think: enableThinking,
        stream: true,
        options: {
          temperature: 0.7,
          num_predict: ollamaPredictNum,
        },
        tools: ollamaEnableTools ? llmTools : undefined,
        keep_alive: -1,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
        responseType: "stream",
      },
    );

    const processLine = async (line: string) => {
      if (line.trim() === "") return;
      try {
        const parsedData = JSON.parse(line);

        // Handle content from Ollama
        if (parsedData.message?.content) {
          const content = parsedData.message.content;
          partialCallback(content);
          partialAnswer += content;
        }

        // Handle thinking from Ollama
        if (parsedData.message?.thinking) {
          const thinking = parsedData.message.thinking;
          partialThinkingCallback?.(thinking);
          partialThinking += thinking;
        }

        // Handle tool calls from Ollama
        if (parsedData.message?.tool_calls) {
          // tool_calls format: [[{"function":{"index":0,"name":"setVolume","arguments":{"percent":50}}}]]
          functionCallsPackages.push(parsedData.message.tool_calls);
        }
      } catch (error) {
        console.error("Error parsing data:", error, line);
      }
    };

    response.data.on("data", async (chunk: Buffer) => {
      buffer += chunk.toString();
      const lines = buffer.split("\n");
      // Keep the last part in buffer as it might be incomplete
      buffer = lines.pop() || "";

      for (const line of lines) {
         await processLine(line);
      }
    });

    response.data.on("end", async () => {
      // Process remaining buffer
      if (buffer.trim() !== "") {
        await processLine(buffer);
      }
      
      console.log("Stream ended");
      const functionCalls = functionCallsPackages.flat().map((call, index) => ({
        id: `call_${Date.now()}_${Math.random()}_${index}`,
        type: "function",
        function: call.function,
      }));
      console.log(
        "functionCallsPackages: ",
        JSON.stringify(functionCallsPackages),
      );
      console.log("functionCalls: ", JSON.stringify(functionCalls));
      messages.push({
        role: "assistant",
        content: partialAnswer,
        tool_calls: functionCallsPackages as any,
      });

      if (!isEmpty(functionCalls)) {
        const results = await Promise.all(
          functionCalls.map(async (call: OllamaFunctionCall) => {
            const {
              function: { arguments: args, name },
            } = call;
            const func = llmFuncMap[name! as string];
            if (func) {
              invokeFunctionCallback?.(name! as string);
              return [
                name,
                await func(args)
                  .then((res) => {
                    invokeFunctionCallback?.(name! as string, res);
                    return res;
                  })
                  .catch((err) => {
                    console.error(`Error executing function ${name}:`, err);
                    return `Error executing function ${name}: ${err.message}`;
                  }),
              ];
            } else {
              console.error(`Function ${name} not found`);
              return [name, `Function ${name} not found`];
            }
          }),
        );

        const newMessages: OllamaMessage[] = results.map(
          ([name, result]: any) => ({
            role: "tool",
            content: result as string,
            tool_name: name as string,
          }),
        );

        // Directly extract and return the tool result if available
        const describeMessage = newMessages.find((msg) =>
          msg.content.startsWith(ToolReturnTag.Response),
        );
        const responseContent = extractToolResponse(
          describeMessage?.content || "",
        );
        if (responseContent) {
          console.log(
            `[LLM] Tool response starts with "[response]", return it directly.`,
          );
          newMessages.push({
            role: "assistant",
            content: responseContent,
          });
          // append responseContent in chunks
          await stimulateStreamResponse({
            content: responseContent,
            partialCallback,
            endResolve,
            endCallback,
          });
          return;
        }

        await chatWithLLMStream(
          newMessages as Message[],
          partialCallback,
          () => {
            endResolve();
            endCallback();
          },
        );
        return;
      } else {
        endResolve();
        endCallback();
      }
    });
  } catch (error: any) {
    console.error("Error:", error.message);
    endResolve();
    endCallback();
  }

  return promise;
};

const summaryTextWithLLM: SummaryTextWithLLMFunction = async (
  text: string, promptPrefix: string
): Promise<string> => {
  const prompt = `${promptPrefix}\n\n${text}\n\n`;

  const response = await axios.post(
    `${ollamaEndpoint}/api/generate`,
    {
      model: ollamaModel,
      prompt: prompt,
      stream: false,
      think: false,
    }
  );

  if (response.data && response.data.response) {
    const summary = response.data.response;
    console.log("Ollama summary:", summary);
    return summary;
  } else {
    console.log("No summary returned from Ollama.");
    return "";
  }
}

const warmupSystemPrompt = async (systemContent: string): Promise<void> => {
  try {
    await axios.post(
      `${ollamaEndpoint}/api/chat`,
      {
        model: ollamaModel,
        messages: [{ role: "system", content: systemContent }],
        options: { num_predict: 0 },
        think: false,
        stream: false,
        keep_alive: -1,
      },
    );
    console.log("[LLM] KV cache warmup done.");
  } catch (err: any) {
    console.error("[LLM] KV cache warmup failed:", err.message);
  }
};

export default { chatWithLLMStream, resetChatHistory, summaryTextWithLLM, warmupSystemPrompt };
