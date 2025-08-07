// import onnxruntimeWeb from "./ort.webgpu.min.mjs"
import onnxruntimeWeb from "./ort.webgpu.min.mjs";

// const MODEL_PATH = "./lg_256_128p.onnx";
const MODEL_PATH = "./superpoint_lightglue_pipeline.onnx";
const INPUT_SIZE = 1024;

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

function createTestElements(key) {
  // Check if elements already exist
  if (document.getElementById("result-" + key)) {
    return; // Elements already exist
  }

  const body = document.body;

  // Create result paragraph
  const resultP = document.createElement("p");
  resultP.id = "result-" + key;
  resultP.textContent = "waiting for " + key;
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
  // Create HTML elements for this test if they don't exist
  createTestElements(key);

  return new Promise(async (resolve, reject) => {
    try {
      const tensor = await getExampleTensor();
      const startTime = performance.now();
      const session = await onnxruntimeWeb.InferenceSession.create(
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
    }
  });
}

async function main() {
  // onnxruntimeWeb.env.logLevel = "verbose";
  // onnxruntimeWeb.env.debug = true;

  await test(
    { executionProviders: [{ name: "webgpu", powerPreference: "low-power" }] },
    "gpu"
  );
  await test(
    {
      executionProviders: [
        {
          name: "webnn",
          deviceType: "gpu",
          powerPreference: "default",
        },
      ],
    },
    "webnn-gpu"
  );
  await test(
    {
      executionProviders: [
        {
          name: "webnn",
          deviceType: "gpu",
          powerPreference: "default",
        },
      ],
    },
    "webnn-gpu"
  );
  await test(
    {
      executionProviders: [
        {
          name: "webnn",
          deviceType: "cpu",
          powerPreference: "default",
        },
      ],
    },
    "webnn-cpu"
  );
  await test(
    {
      executionProviders: [
        {
          name: "webnn",
          deviceType: "npu",
          powerPreference: "default",
        },
      ],
    },
    "webnn-npu"
  );
  await test({ executionProviders: ["wasm"] }, "cpu");
}

main();
