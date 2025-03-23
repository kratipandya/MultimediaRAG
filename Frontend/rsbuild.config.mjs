import { defineConfig } from '@rsbuild/core';
import { pluginReact } from '@rsbuild/plugin-react';
import { pluginSvgr } from '@rsbuild/plugin-svgr';

export default defineConfig({
  plugins: [
    pluginReact(),
    pluginSvgr({
      svgrOptions: { exportType: 'named' },
    }),
  ],
  output: {
    publicPath: '/',
    cleanDistPath: true,
  },
  source: {
    assetsInclude: /\.pdf$/,
    alias: {
      // Ensure paths are correct after project move
      '@': './src',
    },
  },
  tools: {
    postcss: {},
  },
  module: {
    rules: [
      // PDF support
      {
        test: /\.pdf$/,
        type: 'asset/resource',
      },
      // Image support (JPG, PNG, GIF, etc.)
      {
        test: /\.(png|jpe?g|gif)$/i,
        type: 'asset/resource',
      },
      // SVG
      {
        test: /\.svg$/,
        type: 'asset/resource',
        issuer: /\.(css|js|ts|tsx)$/,
      },
      // Audio (MP3, WAV, etc.)
      {
        test: /\.(mp3|wav|ogg)$/,
        type: 'asset/resource',
      },
    ],
  },
});