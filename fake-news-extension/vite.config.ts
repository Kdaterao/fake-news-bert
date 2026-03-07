import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss  from '@tailwindcss/vite';
import webExtension from 'vite-plugin-web-extension';

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte(),
            tailwindcss(),
            webExtension({
          manifest: './manifest.json', // path to your manifest file
        })
  ],
});
