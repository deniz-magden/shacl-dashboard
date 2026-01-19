<template>
  <div class="shape-overview p-4">
    <!-- Loading State -->
    <div v-if="loading" class="text-center py-20">
      <p class="text-gray-600 text-lg">Loading shapes overview...</p>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="text-center py-20">
      <p class="text-red-600 text-lg">{{ error }}</p>
      <button @click="loadOverviewData" class="mt-4 px-6 py-3 bg-blue-500 text-white rounded hover:bg-blue-600">
        Retry
      </button>
    </div>

    <!-- Main Content -->
    <div v-else>
    <!-- Tags Section -->
    <div class="grid grid-cols-4 gap-4 mb-4">
      <div
        v-for="(tag, index) in tags"
        :key="index"
        class="flex flex-row items-center bg-white rounded-lg shadow p-6 hover:shadow-md transition"
      >
        <div class="flex-grow">
          <h3 class="text-sm font-medium text-gray-600 mb-1">{{ tag.title }}</h3>
          <p class="text-3xl font-bold text-gray-800">{{ tag.value }}</p>
        </div>
      </div>
    </div>

    <!-- Plots Section -->
    <div class="grid grid-cols-3 gap-4 mb-4">
      <HistogramChart
        :title="'Distribution of Violations per Constraint'"
        :xAxisLabel="'Number of Violations per Constraint'"
        :yAxisLabel="'Number of Node Shapes'"
        :data="normalizedHistogramViolationData"
        :endpoint="getDistributionSummary"
      />
      <ScatterPlotChart
        :title="'Correlation Between Constraints and Violations'"
        :xAxisLabel="'Number of Constraints'"
        :yAxisLabel="'Violations / Constraint'"
        :data="coveragePlotData"
        :showQuadrants="true"
        :endpoint="getCorrelationSummary"
      />
      <ScatterPlotChart
        :title="'Violation Diversity and Intensity'"
        :xAxisLabel="'Entropy of Constraint Violations'"
        :yAxisLabel="'Violations / Constraints'"
        :data="scatterPlotData"
        :showQuadrants="true"
        :endpoint="getDiversitySummary"
      />
    </div>

    <!-- Table Section -->
    <div class="bg-white border border-gray-200 p-6 rounded-lg shadow-lg">
      <h2 class="text-2xl font-bold text-gray-700 mb-4">Shape Details</h2>
      <table class="w-full border-collapse">
        <thead class="bg-gray-200">
          <tr>
            <th
              v-for="(column, index) in columns"
              :key="index"
              class="text-left px-6 py-3 border-b border-gray-300 text-gray-600 font-medium cursor-pointer"
              @click="sortColumn(column)">
              {{ column.label }}
              <span class="sort-indicator" >
                {{ sortKey === column.field ? (sortOrder === 'asc' ? ' ▲' : ' ▼') : '' }}
              </span>
            </th>
            <th class="text-center px-6 py-3 border-b border-gray-300 text-gray-600 font-medium"></th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="shape in sortedPaginatedData"
            :key="shape.id"
            class="even:bg-gray-50 hover:bg-blue-50 transition-colors"
            @click="goToShape(shape)">
            <td class="px-6 py-4 border-b border-gray-300">{{ shape.name }}</td>
            <td class="px-6 py-4 border-b border-gray-300">{{ shape.violations }}</td>
            <td class="px-6 py-4 border-b border-gray-300">{{ shape.propertyShapes }}</td>
            <td class="px-6 py-4 border-b border-gray-300">{{ shape.focusNodes }}</td>
            <td class="px-6 py-4 border-b border-gray-300">{{ shape.propertyPaths }}</td>
            <td class="px-6 py-4 border-b border-gray-300">{{ shape.mostViolatedConstraint }}</td>
            <td class="px-6 py-4 border-b border-gray-300">{{ shape.violationToConstraintRatio }}</td>
            <td class="px-6 py-4 border-b border-gray-300 text-center">
              <button class="text-blue-600 hover:text-blue-800">
                <font-awesome-icon icon="arrow-right" />
              </button>
            </td>
          </tr>
        </tbody>
      </table>

      <div class="flex justify-between items-center mt-4">
        <button
          :disabled="currentPage === 1"
          @click="prevPage"
          class="px-4 py-2 bg-gray-200 text-gray-600 rounded hover:bg-gray-300 disabled:opacity-50">
          Previous
        </button>
        <span class="text-gray-700">Page {{ currentPage }} of {{ totalPages }}</span>
        <button
          :disabled="currentPage === totalPages"
          @click="nextPage"
          class="px-4 py-2 bg-gray-200 text-gray-600 rounded hover:bg-gray-300 disabled:opacity-50">
          Next
        </button>
      </div>
    </div>
    </div>
  </div>
</template>


<script setup>
/**
 * ShapeOverview component
 *
 * Provides a comprehensive overview of SHACL shapes in the dataset.
 * Displays statistics, visualizations, and listings of shapes with their constraints and validation results.
 *
 * @example
 * // Basic usage in a parent component template:
 * // <ShapeOverview />
 *
 * @prop {Array} [shapes=[]] - List of shapes to display
 * @prop {Boolean} [showViolations=true] - Whether to show violation data
 * @prop {Boolean} [showCharts=true] - Whether to show visualization charts
 *
 * @dependencies
 * - vue (Composition API)
 * - ../Charts/PieChart.vue
 * - ../Charts/GroupedBarChart.vue
 *
 * @style
 * - Responsive layout with cards and data tables.
 * - Data visualization components for shape statistics.
 * - Filterable and sortable shape listings with expandable details.
 *
 * @returns {HTMLElement} A dashboard page showing shape statistics in summary cards at the top,
 * three data visualizations (histogram and scatter plots) in the middle, and a sortable, paginated
 * data table listing all node shapes with their metrics and violation details at the bottom.
 */
// Importing components
import HistogramChart from './../Charts/HistogramChart.vue';
import ScatterPlotChart from './../Charts/ScatterPlotChart.vue';
import { ref, computed, onMounted, watch } from 'vue';
import { useRouter } from 'vue-router';
import {
  getNodeShapesCountInGraph,
  getNodeShapesWithViolationsCountOverview,
  getMaxViolationsForNodeShape,
  getAverageViolationsForNodeShapes,
  getDistributionSummary,
  getCorrelationSummary,
  getDiversitySummary,
  getViolationsDistribution,
  getCorrelationData,
  getNodeShapeDetailsTable
} from '../../services/api.js';
import { usePrefixes } from '../../composables/usePrefixes.js';

// State
const loading = ref(false);
const error = ref(null);

// Use prefixes composable for URI formatting
const { loadPrefixes, formatURI } = usePrefixes();

// Scatter plot data - will be loaded from API
const coveragePlotData = ref({
  datasets: [{
    label: "Shapes",
    data: []
  }]
});

const scatterPlotData = ref({
  datasets: [{
    label: "Shapes",
    data: []
  }]
});


// Router for navigation
const router = useRouter();

// Tags data - will be loaded from API
const tags = ref([
  { title: "Total Node Shapes", value: 0 },
  { title: "Node Shapes with Violations (%)", value: "0%" },
  { title: "Max Violations per Node Shape", value: 0 },
  { title: "Avg Violations per Node Shape", value: 0 },
]);

// Chart data - will be loaded from API
const normalizedHistogramViolationData = ref({
  labels: [],
  datasets: []
});

let summaries = {
  constraint: ref("Loading ..."),
  correlation: ref("Loading ..."),
  diversity: ref("Loading ..."),
}

// Summary options
const summaryLevel = ref("high");
const summaryUseLlm = ref(false);

// Track if summaries are loaded
const summariesLoaded = ref(false);

async function fetchSummary(endpoint, level, useLlm) {
  try {
    const params = new URLSearchParams({
      level,
      use_llm: String(useLlm),
    });
    const response = await fetch(`http://localhost:5000${endpoint}?${params.toString()}`);
    if (!response.ok) {
      return "Error loading summary";
    }
    const data = await response.json();
    return data.summary || "No summary available";
  } catch (error) {
    console.error(`Error fetching summary from ${endpoint}:`, error);
    return "Error loading summary";
  }
}

async function loadSummaries() {
  summariesLoaded.value = false;
  const [constraintSummary, correlationSummary, diversitySummary] = await Promise.all([
    fetchSummary("/summaries/shapes/distribution-constraint", summaryLevel.value, summaryUseLlm.value),
    fetchSummary("/summaries/shapes/correlation", summaryLevel.value, summaryUseLlm.value),
    fetchSummary("/summaries/shapes/diversity-intensity", summaryLevel.value, summaryUseLlm.value),
  ]);

  summaries.constraint.value = constraintSummary;
  summaries.correlation.value = correlationSummary;
  summaries.diversity.value = diversitySummary;
  summariesLoaded.value = true;
}

watch([summaryLevel, summaryUseLlm], loadSummaries);

onMounted(async () => {
  await loadSummaries();
})

// Function to download all summaries as a text file
function downloadAllSummaries() {
  const summaryContent = `SHACL Dashboard - Shape Overview Summary Report
Generated: ${new Date().toLocaleString()}
Level: ${summaryLevel.value}
LLM: ${summaryUseLlm.value ? "on" : "off"}
========================================

1. DISTRIBUTION OF VIOLATIONS PER CONSTRAINT
========================================
${summaries.constraint.value}

2. CORRELATION BETWEEN CONSTRAINTS AND VIOLATIONS
========================================
${summaries.correlation.value}

3. VIOLATION DIVERSITY AND INTENSITY
========================================
${summaries.diversity.value}
`;

  // Create a blob and download
  const blob = new Blob([summaryContent], { type: 'text/plain' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `shacl-shape-overview-summaries-${new Date().toISOString().split('T')[0]}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

const columns = ref([
  { label: "Node Shape Name", field: "name" },
  { label: "Violations", field: "violations" },
  { label: "Number of Property Shapes", field: "propertyShapes" },
  { label: "Focus Nodes Affected", field: "focusNodes" },
  { label: "Property Paths", field: "propertyPaths" },
  { label: "Most Violated Constraint Component", field: "mostViolatedConstraint" },
  { label: "Violation-to-Constraint Ratio", field: "violationToConstraintRatio" },
]);

// Table data - will be loaded from API
const shapes = ref([]);

// Load all overview data from API
const loadOverviewData = async () => {
  loading.value = true;
  error.value = null;

  try {
    // Load prefixes (cached after first call)
    await loadPrefixes();

    // Load tags data in parallel
    const [totalShapesData, shapesWithViolationsData, maxViolationsData, avgViolationsData] = 
      await Promise.all([
        getNodeShapesCountInGraph(),
        getNodeShapesWithViolationsCountOverview(),
        getMaxViolationsForNodeShape(),
        getAverageViolationsForNodeShapes()
      ]);

    // Update tags
    const totalShapes = totalShapesData.nodeShapeCount || 0;
    tags.value[0].value = totalShapes;
    
    const violationsCount = shapesWithViolationsData.nodeShapesWithViolationsCount || 0;
    const percentage = totalShapes > 0 
      ? ((violationsCount / totalShapes) * 100).toFixed(1)
      : 0;
    tags.value[1].value = `${percentage}%`;
    
    tags.value[2].value = maxViolationsData.violationCount || 0;
    tags.value[3].value = avgViolationsData.averageViolations || 0;

    // Load chart data in parallel
    const [histogramData, correlationData, tableData] = await Promise.all([
      getViolationsDistribution(10),
      getCorrelationData(),
      getNodeShapeDetailsTable()
    ]);

    // Update histogram
    normalizedHistogramViolationData.value = histogramData;

    // Update scatter plots from correlation data
    coveragePlotData.value = {
      datasets: [{
        label: "Shapes",
        data: correlationData.map(item => ({
          x: item.num_constraints,
          y: item.num_constraints > 0 ? item.num_violations / item.num_constraints : 0,
          label: "",
          hasZeroViolations: item.num_violations === 0
        }))
      }]
    };

    scatterPlotData.value = {
      datasets: [{
        label: "Shapes",
        data: correlationData.map(item => ({
          x: item.violation_entropy,
          y: item.num_constraints > 0 ? item.num_violations / item.num_constraints : 0,
          label: ""
        }))
      }]
    };

    // Format URIs with prefixes AND store original
    shapes.value = (tableData.nodeShapes || []).map(shape => ({
      ...shape,
      originalName: shape.name,  // Keep full URI for navigation
      name: formatURI(shape.name),  // Display prefixed version
      mostViolatedConstraint: formatURI(shape.mostViolatedConstraint)
    }));

  } catch (err) {
    console.error('Error loading overview data:', err);
    error.value = 'Failed to load overview data. Please try again.';
  } finally {
    loading.value = false;
  }
};
const currentPage = ref(1);
const pageSize = ref(10);
const totalPages = computed(() => Math.ceil(shapes.value.length / pageSize.value));

const paginatedData = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value;
  return shapes.value.slice(start, start + pageSize.value);
});

const sortedPaginatedData = computed(() => {
  const data = paginatedData.value;
  if (sortKey.value) {
    return [...data].sort((a, b) => {
      const result = a[sortKey.value].toString().localeCompare(b[sortKey.value].toString(), undefined, { numeric: true });
      return sortOrder.value === "asc" ? result : -result;
    });
  }
  return data;
});

const prevPage = () => {
  if (currentPage.value > 1) currentPage.value--;
};

const nextPage = () => {
  if (currentPage.value < totalPages.value) currentPage.value++;
};

// Use originalName (full URI) for navigation, not the prefixed name
const goToShape = (shape) => {
  // URL-encode the shape URI to handle special characters and slashes
  const encodedShapeId = encodeURIComponent(shape.originalName);
  router.push({ name: "ShapeView", params: { shapeId: encodedShapeId } });
};

const sortKey = ref("");
const sortOrder = ref("asc");

const sortColumn = (column) => {
  if (sortKey.value === column.field) {
    sortOrder.value = sortOrder.value === "asc" ? "desc" : "asc";
  } else {
    sortKey.value = column.field;
    sortOrder.value = "asc";
  }
};

// Load data on mount
onMounted(() => {
  loadOverviewData();
});
</script>


<style scoped>
.grid-cols-4 {
  grid-template-columns: repeat(4, 1fr);
}

th, td {
  padding: 12px;
}

tbody tr:hover {
  background-color: #f0f8ff;
}

tbody tr {
  cursor: pointer;
}

.grid {
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}

.chart-container {
  height: 100%;
  width: 100%;
  background: white;
  border-radius: 8px;
  padding: 10px;
}

.shape-overview {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.sort-indicator {
  font-size: 0.8em; /* Makes the triangle smaller */
  margin-left: 5px;
  opacity: 0.8; /* Optional: makes it slightly faded */
}
</style>
