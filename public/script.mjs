// import onnxruntimeWeb from "./ort.webgpu.min.mjs"
import onnxruntimeWeb from "./ort.webgpu.min.mjs";

// Model configurations
const MODEL_CONFIGS = {
  "superpoint_lightglue": {
    name: "SuperPoint + LightGlue Pipeline",
    path: "./superpoint_lightglue_pipeline.onnx",
    inputSize: 1024
  },
  "lg_256": {
    name: "LightGlue 256x128p",
    path: "./lg_256_128p.onnx", 
    inputSize: 256
  }
};

// Current model settings (mutable)
let MODEL_PATH = "./superpoint_lightglue_pipeline.onnx";
let INPUT_SIZE = 1024;

function imageToTensor(img) {
  const targetSize = INPUT_SIZE;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to get canvas context");
  }
  // Resize image while maintaining aspect ratio
  const { width, height } = img;
  const scale = Math.min(targetSize / width, targetSize / height);
  const newWidth = Math.round(width * scale);
  const newHeight = Math.round(height * scale);

  canvas.width = targetSize;
  canvas.height = targetSize;

  // Fill with black background
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, targetSize, targetSize);

  // Draw image centered
  const offsetX = (targetSize - newWidth) / 2;
  const offsetY = (targetSize - newHeight) / 2;
  ctx.drawImage(img, offsetX, offsetY, newWidth, newHeight);

  // Get pixel data
  const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
  const { data } = imageData;

  // Convert RGBA to grayscale and normalize to [0, 1]
  const tensorData = new Float32Array(targetSize * targetSize);

  for (let i = 0; i < targetSize * targetSize; i++) {
    const pixelIndex = i * 4; // RGBA

    // Convert to grayscale using luminance formula: 0.299*R + 0.587*G + 0.114*B
    const r = data[pixelIndex] / 255.0;
    const g = data[pixelIndex + 1] / 255.0;
    const b = data[pixelIndex + 2] / 255.0;
    const grayscale = 0.299 * r + 0.587 * g + 0.114 * b;

    tensorData[i] = grayscale;
  }

  // Create tensor with shape [1, 1, height, width] (NCHW format with 1 channel for grayscale)
  return new onnxruntimeWeb.Tensor("float32", tensorData, [
    1,
    1,
    targetSize,
    targetSize,
  ]);
}

export async function getExampleTensor() {
  const imageElement1 = await loadImageFromDataUrl("/img1.jpg");
  const imageTensor1 = imageToTensor(imageElement1);
  const imageElement2 = await loadImageFromDataUrl("/img2.jpg");
  const imageTensor2 = imageToTensor(imageElement2);

  const batchedData = new Float32Array(
    imageTensor1.data.length + imageTensor2.data.length
  );
  batchedData.set(imageTensor1.data, 0);
  batchedData.set(imageTensor2.data, imageTensor1.data.length);

  return new onnxruntimeWeb.Tensor("float32", batchedData, [
    2,
    1,
    INPUT_SIZE,
    INPUT_SIZE,
  ]);
}

function loadImageFromDataUrl(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = dataUrl;
  });
}

function updateModelSettings(modelKey) {
  const config = MODEL_CONFIGS[modelKey];
  if (config) {
    MODEL_PATH = config.path;
    INPUT_SIZE = config.inputSize;
    console.log(`Switched to model: ${config.name}, Input size: ${INPUT_SIZE}`);
    
    // Update display to show current model
    const modelDisplay = document.getElementById("current-model-display");
    if (modelDisplay) {
      modelDisplay.textContent = `Current: ${config.name} (${INPUT_SIZE}x${INPUT_SIZE})`;
    }
  }
}

function createTestElements(key, displayName) {
  // Check if elements already exist
  if (document.getElementById("result-" + key)) {
    return; // Elements already exist
  }

  const body = document.body;

  // Create button for this test
  const button = document.createElement("button");
  button.id = "button-" + key;
  button.textContent = "Run " + displayName;
  button.style.marginRight = "10px";
  button.style.marginBottom = "10px";
  body.appendChild(button);

  // Create result paragraph
  const resultP = document.createElement("p");
  resultP.id = "result-" + key;
  resultP.textContent = "Click button to run " + displayName;
  body.appendChild(resultP);

  // Create error paragraph
  const errorP = document.createElement("p");
  errorP.id = "error-" + key;
  errorP.textContent = "no error";
  body.appendChild(errorP);

  // Add line break
  const br = document.createElement("br");
  body.appendChild(br);
}

async function test(sessionParams, key) {
  return new Promise(async (resolve) => {
    let session = null;
    try {
      const tensor = await getExampleTensor();
      const startTime = performance.now();
      document.getElementById("result-" + key).innerText =
        "loading " + key + "...";
      session = await onnxruntimeWeb.InferenceSession.create(
        MODEL_PATH,
        sessionParams
      );
      const runStartTime = performance.now();
      const loadTime = (runStartTime - startTime).toFixed(2);
      document.getElementById("result-" + key).innerText =
        "running " + key + ", loaded in " + loadTime + "ms";

      const result = await session.run({ images: tensor });
      const runEndTime = performance.now();

      document.getElementById("result-" + key).innerText =
        key +
        ": " +
        result.matches.size / 3 +
        " took " +
        (runEndTime - runStartTime).toFixed(2) +
        "ms, loaded in " +
        loadTime +
        "ms";
      resolve();
    } catch (error) {
      document.getElementById("error-" + key).innerText = error.toString();
      resolve();
    } finally {
        if (session && session.handler && session.handler.dispose) {
            console.log("disposing session");
            session.handler.dispose();
        }
    }
  });
}

function getTestConfigurations() {
  return [
    {
      key: "webgpu",
      displayName: "WebGPU",
      sessionParams: { executionProviders: [{ name: "webgpu", powerPreference: "low-power" }] }
    },
    {
      key: "webnn-gpu",
      displayName: "WebNN GPU",
      sessionParams: {
        executionProviders: [
          {
            name: "webnn",
            deviceType: "gpu",
            powerPreference: "default",
          },
        ],
      }
    },
    {
      key: "webnn-cpu",
      displayName: "WebNN CPU",
      sessionParams: {
        executionProviders: [
          {
            name: "webnn",
            deviceType: "cpu",
            powerPreference: "default",
          },
        ],
      }
    },
    {
      key: "webnn-npu",
      displayName: "WebNN NPU",
      sessionParams: {
        executionProviders: [
          {
            name: "webnn",
            deviceType: "npu",
            powerPreference: "default",
          },
        ],
      }
    },
    {
      key: "cpu",
      displayName: "WASM CPU",
      sessionParams: { executionProviders: ["wasm"] }
    }
  ];
}

async function main() {
  // onnxruntimeWeb.env.logLevel = "verbose";
  // onnxruntimeWeb.env.debug = true;
  
  // Thread count management
  let currentThreadCount = 8;
  onnxruntimeWeb.env.wasm.numThreads = currentThreadCount;
  onnxruntimeWeb.env.wasm.proxy = true;

  // Create model selection dropdown
  const modelSelectorLabel = document.createElement("label");
  modelSelectorLabel.textContent = "Select Model: ";
  modelSelectorLabel.style.fontWeight = "bold";
  modelSelectorLabel.style.marginRight = "10px";
  document.body.appendChild(modelSelectorLabel);

  const modelSelector = document.createElement("select");
  modelSelector.id = "model-selector";
  modelSelector.style.marginRight = "20px";
  modelSelector.style.marginBottom = "10px";
  modelSelector.style.padding = "5px";

  // Add options to dropdown
  Object.keys(MODEL_CONFIGS).forEach(key => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = MODEL_CONFIGS[key].name;
    if (key === "superpoint_lightglue") { // Set default selection
      option.selected = true;
    }
    modelSelector.appendChild(option);
  });
  
  document.body.appendChild(modelSelector);

  // Create current model display
  const modelDisplay = document.createElement("span");
  modelDisplay.id = "current-model-display";
  modelDisplay.style.marginLeft = "10px";
  modelDisplay.style.fontStyle = "italic";
  modelDisplay.style.color = "#666";
  document.body.appendChild(modelDisplay);
  
  // Initialize display
  updateModelSettings("superpoint_lightglue");

  // Add line break after model selector
  const brAfterModel = document.createElement("br");
  document.body.appendChild(brAfterModel);
  
  // Create WASM Thread Count Input
  const threadInputContainer = document.createElement("div");
  threadInputContainer.style.marginTop = "10px";
  threadInputContainer.style.marginBottom = "10px";
  
  const threadInputLabel = document.createElement("label");
  threadInputLabel.textContent = "WASM Threads: (refresh page before changing):";
  threadInputLabel.style.fontWeight = "bold";
  threadInputLabel.style.marginRight = "10px";
  threadInputContainer.appendChild(threadInputLabel);
  
  const threadInput = document.createElement("input");
  threadInput.type = "number";
  threadInput.id = "thread-input";
  threadInput.value = currentThreadCount;
  threadInput.min = "1";
  threadInput.max = "16";
  threadInput.style.width = "60px";
  threadInput.style.marginRight = "10px";
  threadInput.style.padding = "2px 5px";
  threadInputContainer.appendChild(threadInput);
  
  const hardwareConcurrencyText = document.createElement("span");
  hardwareConcurrencyText.textContent = `(Hardware concurrency: ${navigator.hardwareConcurrency || 'unknown'})`;
  hardwareConcurrencyText.style.fontStyle = "italic";
  hardwareConcurrencyText.style.color = "#666";
  threadInputContainer.appendChild(hardwareConcurrencyText);
  
  document.body.appendChild(threadInputContainer);
  
  // Add line break for spacing
  const brSpacing = document.createElement("br");
  document.body.appendChild(brSpacing);

  const testConfigs = getTestConfigurations();
  
  // Create "Run All Tests" button
  const runAllButton = document.createElement("button");
  runAllButton.textContent = "Run All Tests";
  runAllButton.style.marginRight = "20px";
  runAllButton.style.marginBottom = "20px";
  runAllButton.style.fontWeight = "bold";
  runAllButton.style.padding = "10px 20px";
  document.body.appendChild(runAllButton);
  
  // Add line break after "Run All" button
  const brAfterRunAll = document.createElement("br");
  document.body.appendChild(brAfterRunAll);
  
  // Create UI elements for all tests
  testConfigs.forEach(config => {
    createTestElements(config.key, config.displayName);
    
    // Add click event listener to the button
    const button = document.getElementById("button-" + config.key);
    button.addEventListener("click", async () => {
      button.disabled = true;
      button.textContent = "Running...";
      await test(config.sessionParams, config.key);
      button.disabled = false;
      button.textContent = "Run " + config.displayName;
    });
  });
  
  // Add change event listener to model selector dropdown
  modelSelector.addEventListener("change", (event) => {
    updateModelSettings(event.target.value);
  });

  // Add input event listener for real-time updates and change event for validation
  const handleThreadCountUpdate = (event) => {
    const newThreadCount = parseInt(event.target.value);
    
    // Validate the input
    if (newThreadCount >= 1 && newThreadCount <= 16) {
      currentThreadCount = newThreadCount;
      // Update ONNX Runtime WASM thread count
      onnxruntimeWeb.env.wasm.numThreads = currentThreadCount;
      console.log(`WASM thread count updated to: ${currentThreadCount}`);
    } else if (event.type === "change") {
      // Only reset on blur/change, not while typing
      event.target.value = currentThreadCount;
      console.warn(`Invalid thread count: ${newThreadCount}. Must be between 1 and 16.`);
    }
  };
  
  threadInput.addEventListener("input", handleThreadCountUpdate);
  threadInput.addEventListener("change", handleThreadCountUpdate);

  // Add click event listener to "Run All Tests" button
  runAllButton.addEventListener("click", async () => {
    runAllButton.disabled = true;
    runAllButton.textContent = "Running All Tests...";
    
    // Disable all individual buttons
    testConfigs.forEach(config => {
      const button = document.getElementById("button-" + config.key);
      button.disabled = true;
    });
    
    // Run all tests sequentially
    for (const config of testConfigs) {
      await test(config.sessionParams, config.key);
    }
    
    // Re-enable all buttons
    testConfigs.forEach(config => {
      const button = document.getElementById("button-" + config.key);
      button.disabled = false;
    });
    
    runAllButton.disabled = false;
    runAllButton.textContent = "Run All Tests";
  });
}

main();
