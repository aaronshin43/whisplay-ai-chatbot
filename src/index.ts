import { display } from "./device/display";
import Battery from "./device/battery";
import ChatFlow from "./core/ChatFlow";
import dotenv from "dotenv";
import { initializeMatcher } from "./cloud-api/local/oasis-matcher-node";

dotenv.config();

// Pre-load OASIS Matcher in the background
if (process.env.ENABLE_OASIS_MATCHER !== "false") {
  initializeMatcher().catch(e => console.error("Failed to initialize OASIS Matcher:", e));
}

const battery = new Battery();
battery.connect().catch(e => {
  console.error("fail to connect to battery service:", e);
});
battery.addListener("batteryLevel", (data: any) => {
  display({
    battery_level: data,
  });
});

new ChatFlow({
  enableCamera: process.env.ENABLE_CAMERA === "true",
});
