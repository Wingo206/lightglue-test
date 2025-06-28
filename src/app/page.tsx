"use client";

import { env, InferenceSession } from "onnxruntime-web/webgpu";
// import { env, InferenceSession } from "onnxruntime-web";
import { useState } from "react";
import { getExampleTensor } from "./getImage";

export default function Home() {
  const [session, setSession] = useState<InferenceSession | null>(null);

  async function createInferenceSession(enableWebGPU: boolean) {
    env.wasm.wasmPaths = "/"
    env.wasm.numThreads = navigator.hardwareConcurrency;

    // env.logLevel = "verbose";
    // env.debug = true;
    
    try {
      const session = await InferenceSession.create("/lg_256_128p.onnx", {
        executionProviders: enableWebGPU ? ["webgpu"] : ["wasm"],
      });
      setSession(session);
      console.log("Session created successfully with", enableWebGPU ? "WebGPU" : "WASM");
    } catch (error) {
      console.error("Failed to create session:", error);
    }
  }

  async function runModel() {
    if (!session) {
      console.log("No session available");
      return;
    }
    
    try {
      const inputTensor = await getExampleTensor();
      const result = await session.run({ images: inputTensor });

      console.log(result, "Matches: ", result.matches.size / 3);

    } catch (error) {
      console.error("Failed to run model:", error);
    }
  }

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
        <div className="flex flex-col gap-4">
          <button 
            className="bg-gray-500 text-white p-2 rounded-md hover:bg-gray-600"
            onClick={() => createInferenceSession(false)}
          >
            Create session without GPU (WASM)
          </button>
          <button 
            className="bg-green-500 text-white p-2 rounded-md hover:bg-green-600"
            onClick={() => createInferenceSession(true)}
          >
            Create session with GPU (WebGPU)
          </button>
          <button
            className="bg-blue-500 text-white p-2 rounded-md hover:bg-blue-600 disabled:bg-gray-400"
            onClick={() => runModel()}
            disabled={!session}
          >
            Run model {!session && "(Create session first)"}
          </button>
        </div>
      </main>
    </div>
  );
}
