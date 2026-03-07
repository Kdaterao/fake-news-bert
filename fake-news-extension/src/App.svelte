<script lang="ts">
import "./app.css"
import { checkText }  from "./lib/client"


    type ApiResult = {
      prediction: boolean;
      confidence: number;
      links?: string[];
      error?: string;
    };

    let text = "";
    let result: ApiResult | null = null;
    let loading = false;
    

    async function handleCheck() {
      loading = true;
      result = await checkText(text);
      loading = false;
    }

</script>

<main class="p-4 w-80 font-sans">
  <h1 class="text-lg font-bold mb-2 text-center">Fake News Checker</h1>

  <div class="flex flex-col gap-2">
    <textarea
      bind:value={text}
      placeholder="Paste news text here..."
      class="border rounded-md p-2 resize-none h-24 focus:outline-none focus:ring-2 focus:ring-blue-500"
    ></textarea>

    <button
      on:click={handleCheck}
      class="bg-blue-600 text-white font-semibold py-2 rounded-md hover:bg-blue-700 transition disabled:bg-gray-400"
      disabled={loading}
    >
      {#if loading}Checking...{:else}Check{/if}
    </button>
  </div>

  {#if result}
    <div class="mt-4 p-2 border rounded-md bg-gray-50">
      {#if result.error}
        <p class="text-red-600 font-medium">{result.error}</p>
      {:else}
        <p class="font-semibold">Prediction: <span class="text-blue-700">{result.prediction ? "True" : "False"}</span></p>
        <p class="font-medium">Confidence: <span class="text-gray-800">{result.confidence.toFixed(2)}</span></p>

        {#if result.links && result.links.length > 0}
          <p class="mt-2 font-medium">Links:</p>
          <ul class="list-disc list-inside ml-2 text-blue-600">
            {#each result.links as link}
              <li><a href={link} target="_blank" class="hover:underline">{link}</a></li>
            {/each}
          </ul>
        {/if}
      {/if}
    </div>
  {/if}
</main>
<style global lang="postcss">
  @reference "tailwindcss";
</style>
