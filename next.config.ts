import type { NextConfig } from "next";
import { webpack } from "next/dist/compiled/webpack/webpack";
import CopyWebpackPlugin from "copy-webpack-plugin";

const nextConfig: NextConfig = {
  webpack: (config, { isServer }) => {
    // Configure ONNX runtime for client-side
    // if (!isServer) {
    //   config.resolve.fallback = {
    //     ...config.resolve.fallback,
    //     fs: false,
    //     path: false,
    //     crypto: false,
    //   };

    //   // Externalize ONNX runtime to prevent bundling issues
    //   config.externals = config.externals || [];
    // }
    config.watchOptions.ignored = ["**/node_modules", "**/node_modules/**"];
    config.plugins.push(
      new CopyWebpackPlugin({
        patterns: [
          {
            from: "./node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs",
            to: "../public/ort-wasm-simd-threaded.mjs",
          },
          {
            from: "./node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm",
            to: "../public/ort-wasm-simd-threaded.wasm",
          },
          {
            from: "./node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs",
            to: "../public/ort-wasm-simd-threaded.jsep.mjs",
          },
          {
            from: "./node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm",
            to: "../public/ort-wasm-simd-threaded.jsep.wasm",
          },
        ],
      }),
    );

    // Handle .wasm files
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      layers: true,
    };

    // // Handle .mjs files from ONNX runtime
    // config.module.rules.push({
    //   test: /\.mjs$/,
    //   include: /node_modules/,
    //   type: "javascript/auto",
    // });

    // // Ignore ONNX runtime files from being processed
    // config.module.rules.push({
    //   test: /ort\..*\.mjs$/,
    //   type: "javascript/auto",
    // });

    return config;
  },
  // Allow serving WASM and ONNX files
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "Cross-Origin-Embedder-Policy",
            value: "require-corp",
          },
          {
            key: "Cross-Origin-Opener-Policy",
            value: "same-origin",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
