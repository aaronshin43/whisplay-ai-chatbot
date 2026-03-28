import moment from "moment";
import {
  getCurrentTimeTag,
  getRecordFileDurationMs,
  splitSentences,
} from "./../utils/index";
import { compact, get, noop } from "lodash";
import {
  onButtonPressed,
  onButtonReleased,
  onButtonDoubleClick,
  display,
  getCurrentStatus,
  onCameraCapture,
} from "../device/display";
import { recordAudioManually, recordFileFormat } from "../device/audio";
import {
  recognizeAudio,
  chatWithLLMStream,
  ttsProcessor,
} from "../cloud-api/server";
import { extractEmojis } from "../utils";
import { StreamResponser } from "./StreamResponsor";
import { cameraDir, recordingsDir } from "../utils/dir";
import { getLatestDisplayImg, setLatestCapturedImg } from "../utils/image";
import dotEnv from "dotenv";
import { getSystemPromptWithKnowledge } from "./Knowledge";
import { getSystemPromptFromOasis, sanitizeOasisChunk, logOasisResponse, dispatchQuery } from "./OasisAdapter";
import { enableRAG } from "../cloud-api/knowledge";

// Use OASIS classify dispatch instead of generic RAG if enabled
const useOasis = process.env.ENABLE_OASIS_MATCHER === "true" || true; // Default to true for now as per request

// Triage hint TTL — must match TRIAGE_HINT_TTL_SEC in python/oasis-classify/config.py
const TRIAGE_HINT_TTL_MS = 60_000;  // 60 seconds, matches Python TRIAGE_HINT_TTL_SEC


dotEnv.config();

class ChatFlow {
  currentFlowName: string = "";
  recordingsDir: string = "";
  currentRecordFilePath: string = "";
  asrText: string = "";
  streamResponser: StreamResponser;
  partialThinking: string = "";
  thinkingSentences: string[] = [];
  answerId: number = 0;
  enableCamera: boolean = false;
  knowledgePrompts: string[] = [];
  private triageHint: { category: string; expiresAt: number } | null = null;

  constructor(options: { enableCamera?: boolean } = {}) {
    console.log(`[${getCurrentTimeTag()}] ChatBot started.`);
    this.recordingsDir = recordingsDir;
    this.setCurrentFlow("sleep");
    this.streamResponser = new StreamResponser(
      ttsProcessor,
      (sentences: string[]) => {
        if (this.currentFlowName !== "answer") return;
        const fullText = sentences.join(" ");
        display({
          status: "answering",
          emoji: extractEmojis(fullText) || "😊",
          text: fullText,
          RGB: "#0000ff",
          scroll_speed: 3,
        });
      },
      (text: string) => {
        if (this.currentFlowName !== "answer") return;
        display({
          status: "answering",
          text: text || undefined,
          scroll_speed: 3,
        });
      },
    );
    if (options?.enableCamera) {
      this.enableCamera = true;
    }
  }

  async recognizeAudio(path: string): Promise<string> {
    if ((await getRecordFileDurationMs(path)) < 500) {
      console.log("Record audio too short, skipping recognition.");
      return Promise.resolve("");
    }
    console.time(`[ASR time]`);
    const result = await recognizeAudio(path);
    console.timeEnd(`[ASR time]`);
    return result;
  }

  partialThinkingCallback = (partialThinking: string): void => {
    this.partialThinking += partialThinking;
    const { sentences, remaining } = splitSentences(this.partialThinking);
    if (sentences.length > 0) {
      this.thinkingSentences.push(...sentences);
      const displayText = this.thinkingSentences.join(" ");
      display({
        status: "Thinking",
        emoji: "🤔",
        text: displayText,
        RGB: "#ff6800", // yellow
        scroll_speed: 6,
      });
    }
    this.partialThinking = remaining;
  };

  private getActiveTriageHint(): string | null {
    if (!this.triageHint) return null;
    if (Date.now() > this.triageHint.expiresAt) {
      this.triageHint = null;
      return null;
    }
    return this.triageHint.category;
  }

  setCurrentFlow = (flowName: string): void => {
    console.log(`[${getCurrentTimeTag()}] switch to:`, flowName);
    switch (flowName) {
      case "sleep":
        this.currentFlowName = "sleep";
        onButtonPressed(() => {
          this.setCurrentFlow("listening");
        });
        onButtonReleased(noop);
        // camera mode
        if (this.enableCamera) {
          const captureImgPath = `${cameraDir}/capture-${moment().format(
            "YYYYMMDD-HHmmss",
          )}.jpg`;
          onButtonDoubleClick(() => {
            display({
              camera_mode: true,
              capture_image_path: captureImgPath,
            });
          });
          onCameraCapture(() => {
            setLatestCapturedImg(captureImgPath);
          });
        }
        display({
          status: "idle",
          emoji: "😴",
          RGB: "#000055",
          ...(getCurrentStatus().text === "Listening..."
            ? {
                text: `Long Press the button to say something${
                  this.enableCamera ? ",\ndouble click to launch camera" : ""
                }.`,
              }
            : {}),
        });
        break;
      case "listening":
        this.answerId += 1;
        this.currentFlowName = "listening";
        this.currentRecordFilePath = `${
          this.recordingsDir
        }/user-${Date.now()}.${recordFileFormat}`;
        onButtonPressed(noop);
        const { result, stop } = recordAudioManually(
          this.currentRecordFilePath,
        );
        onButtonReleased(() => {
          stop();
          display({
            RGB: "#ff6800", // yellow
          });
        });
        result
          .then(() => {
            this.setCurrentFlow("asr");
          })
          .catch((err) => {
            console.error("Error during recording:", err);
            this.setCurrentFlow("sleep");
          });
        display({
          status: "listening",
          emoji: "😐",
          RGB: "#00ff00",
          text: "Listening...",
        });
        break;
      case "asr":
        this.currentFlowName = "asr";
        display({
          status: "recognizing",
        });
        onButtonDoubleClick(null);
        Promise.race([
          this.recognizeAudio(this.currentRecordFilePath),
          new Promise<string>((resolve) => {
            onButtonPressed(() => {
              resolve("[UserPress]");
            });
            onButtonReleased(noop);
          }),
        ]).then((result) => {
          if (this.currentFlowName !== "asr") return;
          if (result === "[UserPress]") {
            this.setCurrentFlow("listening");
          } else {
            if (result) {
              console.log("Audio recognized result:", result);
              this.asrText = result;
              display({ status: "recognizing", text: result });
              this.setCurrentFlow("answer");
            } else {
              this.setCurrentFlow("sleep");
            }
          }
        });
        break;
      case "answer":
        display({
          status: "answering...",
          RGB: "#00c8a3",
        });
        this.currentFlowName = "answer";
        const currentAnswerId = this.answerId;
        onButtonPressed(() => {
          this.setCurrentFlow("listening");
        });
        onButtonReleased(noop);
        const {
          partial,
          endPartial,
          getPlayEndPromise,
          stop: stopPlaying,
        } = this.streamResponser;
        this.partialThinking = "";
        this.thinkingSentences = [];

        if (useOasis) {
          // ── OASIS classify dispatch path ───────────────────────────────────
          const capturedQuery = this.asrText;

          dispatchQuery(capturedQuery, this.getActiveTriageHint())
            .then((dispatch) => {
              if (currentAnswerId !== this.answerId) return;

              if (dispatch.mode === "direct_response" || dispatch.mode === "ood_response") {
                // Pre-baked response — speak directly, no LLM call
                this.triageHint = null;
                logOasisResponse(capturedQuery, dispatch.responseText);
                partial(dispatch.responseText);
                endPartial();

              } else if (dispatch.mode === "llm_prompt") {
                // Classifier hit — compact prompt → LLM
                this.triageHint = null;
                const prompt: { role: "system" | "user"; content: string }[] = [
                  { role: "system", content: dispatch.systemPrompt },
                  { role: "user",   content: capturedQuery },
                ];

                const oasisBuffer: string[] = [];

                chatWithLLMStream(
                  prompt,
                  (text) => {
                    if (currentAnswerId !== this.answerId) return;
                    const clean = sanitizeOasisChunk(text);
                    oasisBuffer.push(clean);
                    partial(clean);
                  },
                  () => {
                    if (currentAnswerId !== this.answerId) return;
                    const fullResponse = oasisBuffer.join("");
                    logOasisResponse(capturedQuery, fullResponse);
                    console.log("[OASIS] Full response:\n", fullResponse);
                    endPartial();
                  },
                  (partialThinking) =>
                    currentAnswerId === this.answerId &&
                    this.partialThinkingCallback(partialThinking),
                  (functionName: string, result?: string) => {
                    if (result) {
                      display({ text: `[${functionName}]${result}` });
                    } else {
                      display({ text: `Invoking [${functionName}]...` });
                    }
                  },
                );

              } else {
                // triage_prompt — LLM asks clarifying question; store hint with TTL
                this.triageHint = {
                  category:  dispatch.triageHint,
                  expiresAt: Date.now() + TRIAGE_HINT_TTL_MS,
                };
                if (dispatch.raw.hint_changed_result) {
                  console.log(`[Classify] hint changed result for category=${dispatch.raw.category}`);
                }
                const prompt: { role: "system" | "user"; content: string }[] = [
                  { role: "system", content: dispatch.systemPrompt },
                  { role: "user",   content: capturedQuery },
                ];

                const oasisBuffer: string[] = [];

                chatWithLLMStream(
                  prompt,
                  (text) => {
                    if (currentAnswerId !== this.answerId) return;
                    const clean = sanitizeOasisChunk(text);
                    oasisBuffer.push(clean);
                    partial(clean);
                  },
                  () => {
                    if (currentAnswerId !== this.answerId) return;
                    const fullResponse = oasisBuffer.join("");
                    logOasisResponse(capturedQuery, fullResponse);
                    console.log("[OASIS] Full response:\n", fullResponse);
                    endPartial();
                  },
                  (partialThinking) =>
                    currentAnswerId === this.answerId &&
                    this.partialThinkingCallback(partialThinking),
                  (functionName: string, result?: string) => {
                    if (result) {
                      display({ text: `[${functionName}]${result}` });
                    } else {
                      display({ text: `Invoking [${functionName}]...` });
                    }
                  },
                );
              }
            });

        } else {
          // ── Legacy RAG / knowledge path ────────────────────────────────────
          let systemPromptPromise = Promise.resolve("");
          if (enableRAG) {
            systemPromptPromise = getSystemPromptWithKnowledge(this.asrText);
          }

          systemPromptPromise
            .then((res: string) => {
              let knowledgePrompt = res;
              if (res) {
                console.log("Retrieved knowledge for RAG:\n", res);
              }
              if (this.knowledgePrompts.includes(res)) {
                console.log(
                  "[RAG] Knowledge prompt already used in this session, skipping to avoid repetition.",
                );
                knowledgePrompt = "";
              }
              if (knowledgePrompt) {
                this.knowledgePrompts.push(knowledgePrompt);
              }
              const prompt: {
                role: "system" | "user";
                content: string;
              }[] = compact([
                knowledgePrompt
                  ? {
                      role: "system" as const,
                      content: knowledgePrompt,
                    }
                  : null,
                {
                  role: "user" as const,
                  content: this.asrText,
                },
              ]);

              chatWithLLMStream(
                prompt,
                (text) => {
                  if (currentAnswerId !== this.answerId) return;
                  partial(text);
                },
                () => {
                  if (currentAnswerId !== this.answerId) return;
                  endPartial();
                },
                (partialThinking) =>
                  currentAnswerId === this.answerId &&
                  this.partialThinkingCallback(partialThinking),
                (functionName: string, result?: string) => {
                  if (result) {
                    display({
                      text: `[${functionName}]${result}`,
                    });
                  } else {
                    display({
                      text: `Invoking [${functionName}]...`,
                    });
                  }
                },
              );
            });
        }
        getPlayEndPromise().then(() => {
          if (this.currentFlowName === "answer") {
            const img = getLatestDisplayImg();
            if (img) {
              display({
                image: img,
              });
              this.setCurrentFlow("image");
            } else {
              this.setCurrentFlow("sleep");
            }
          }
        });
        onButtonPressed(() => {
          stopPlaying();
          this.setCurrentFlow("listening");
        });
        onButtonReleased(noop);
        break;
      case "image":
        onButtonPressed(() => {
          display({ image: "" });
          this.setCurrentFlow("listening");
        });
        onButtonReleased(noop);
        break;
      default:
        console.error("Unknown flow name:", flowName);
        break;
    }
  };
}

export default ChatFlow;
