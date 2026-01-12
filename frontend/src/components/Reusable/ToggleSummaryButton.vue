<template>
  <div class="relative group inline-block z-50" @mouseenter="adjustPosition">
    <!-- Blue Circle with Question Mark -->
    <div class="w-6 h-6 bg-gray-500 text-white flex items-center justify-center rounded-full cursor-pointer" @click="openDialog">
      ?
    </div>
    <!-- Tooltip -->
    <div
      ref="tooltip"
      class="absolute hidden group-hover:block bg-gray-100 text-gray-800 text-sm p-2 rounded shadow-md w-96 mt-2 right-0 z-50 transition-transform"
      :class="tooltipClass"
    >
      <p class="line-clamp-2 whitespace-pre-line">
        {{ explanation }}
      </p>
      <span class="text-xs text-gray-500">Click to expand</span>
    </div>
  </div>

  <!-- Summary Popup Dialog -->
  <dialog
    ref="dialog"
    class="fixed m-auto rounded-lg p-6 w-[600px] max-w-[90vw] backdrop:bg-black/40"
  >
    <div class="flex justify-between items-start mb-4">
      <h3 class="text-lg font-semibold">Summary</h3>
      <button class="text-gray-500 hover:text-black" @click="closeDialog">X</button>
    </div>

    <!-- Summary Options -->
    <div class="flex gap-4 mb-4 text-sm">
      <label class="flex items-center gap-2">
        <input type="checkbox" @click="level = (level === 'high' ? 'low' : 'high')">
        Detailed
      </label>

      <label class="flex items-center gap-2">
        <input type="checkbox" @click="use_llm = !use_llm">
        LLM Enhancement
      </label>
    </div>

    <!-- Summary Text -->
    <div class="max-h-[60vh] whitespace-pre-line text-sm">
      {{explanation}}
    </div>

    <div class="mt-4 text-right">
      <button class="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300" @click="closeDialog">Close</button>
    </div>
  </dialog>
</template>

<script setup>
/**
 * ToggleSummaryButton component
 *
 * Renders a question mark icon with a hover tooltip for providing additional information.
 * Clickable for Summary Dialog Window.
 *
 * @example
 * // Basic usage in a parent component template:
 * // <ToggleSummaryButton explanation="First 2 Lines of Summary /n Click to expand" />
 *
 * @prop {Function} endpoint - Endpoint to backend Summary Generation
 *
 * @dependencies
 * - vue (Composition API)
 *
 * @style
 * - Round gray background with white question mark
 * - Tooltip appears on hover with readable text
 * - Z-index handling for proper layering
 *
 * @returns {HTMLElement} A circular question mark icon with a hover-triggered tooltip
 * that displays the first 2 lines of an explanation text provided via an endpoint, positioned to the right of the icon.
 * Clickable for an additional Dialog Window with full Summary and further options.
 */
import { defineProps, ref, watch, onMounted } from 'vue';

// Define props to allow customization of explanation text
const props = defineProps({
  endpoint: {
    type: Function,
    required: true
  },
});

// Relocate tooltip to stay inside the window
const tooltip = ref(null);
const tooltipClass = ref('translate-x-full');

function adjustPosition() {
  const rect = tooltip.value?.getBoundingClientRect();
  if (rect.right > window.innerWidth) {
    tooltipClass.value = 'translate-x-2';
  }
}

// Fetch Summaries
const level = ref('high');                  // high = simple, low = detailed
const use_llm = ref(false);
const explanation = ref('Loading ...');

watch([level, use_llm], loadSummary);

onMounted(async () => {
  await loadSummary()
});

async function loadSummary() {
  explanation.value = "Loading ...";
  const json = await props.endpoint(level.value, use_llm.value);
  explanation.value = json.summary.toString();
}

// Dialog window functionality
const dialog = ref(null);

function openDialog() {
  dialog.value?.showModal();
}

function closeDialog() {
  dialog.value?.close();
}
</script>

<style scoped>
/* Optional: Customize the tooltip's position and animation */
.group:hover .hidden {
  display: block;
}
</style>
