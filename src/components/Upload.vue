<template>
  <div class="upload-root">
    <div :class="dropBoxClass" @click="handle">
      <InboxOutlined class="drop-icon" />
      <span class="drop-title">{{ title }}</span>
      <div class="drop-subtitle" v-html="subtitle"></div>
    </div>
  </div>
</template>
<script setup lang="ts">
import { ref, onBeforeUnmount } from "vue";
import { InboxOutlined } from "@ant-design/icons-vue";

import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { useVModels } from "@vueuse/core";

const props = defineProps({
  title: { type: String, required: true },
  subtitle: { type: String, required: false },
  isDirectory: { type: Boolean, default: false },
  isMultiple: { type: Boolean, default: false },
});
const emit = defineEmits<{
  (e: "open", path: string | string[]): void;
  (e: "drop", path: string | string[]): void;
  (e: "dropHoverDrop", {}): void;
  (e: "dropCancelled", {}): void;
}>();
const { title, subtitle, isDirectory, isMultiple } = useVModels(props, emit);
const dropBoxClass = ref("dropBox");
const handle = async () => {
  const selected = await open({
    directory: isDirectory.value,
    multiple: isMultiple.value,
  });
  if (selected !== null) {
    emit("open", selected);
  }
};

const drop = await listen("tauri://file-drop", (event: any) => {
  dropBoxClass.value = "dropBox";
  if (event.payload.length == 1 && !isMultiple) {
    emit("drop", event.payload[0]);
  } else {
    emit("drop", event.payload);
  }
});

const dropHover = await listen("tauri://file-drop-hover", (event: any) => {
  dropBoxClass.value = "dropBoxHover";
  emit("dropHoverDrop", "ok");
});

const dropCancelled = await listen("tauri://file-drop-cancelled", () => {
  dropBoxClass.value = "dropBox";
  emit("dropCancelled", "ok");
});

onBeforeUnmount(() => {
  drop();
  dropHover();
  dropCancelled();
});
</script>

<style scoped>
.upload-root {
  width: 100%;
}

.dropBox {
  width: 100%;
  min-height: 130px;
  border: 1px dashed var(--panel-border);
  border-radius: 10px;
  background: var(--panel-bg);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 14px 12px;
  box-sizing: border-box;
  transition: border-color 0.2s ease, background-color 0.2s ease;
}

.dropBox:hover {
  border: 1px dashed var(--accent);
  background: var(--panel-bg-strong);
  cursor: pointer;
}

.dropBoxHover {
  width: 100%;
  min-height: 130px;
  border: 1px dashed var(--accent);
  border-radius: 10px;
  background: var(--panel-bg-strong);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 14px 12px;
  box-sizing: border-box;
  transition: border-color 0.2s ease, background-color 0.2s ease;
}

.drop-icon {
  font-size: 34px;
  color: var(--accent);
}

.drop-title {
  display: block;
  font-size: 16px;
  line-height: 1.35;
  font-weight: 600;
  color: var(--text-primary);
  text-align: center;
}

.drop-subtitle {
  font-size: 13px;
  line-height: 1.55;
  color: var(--text-secondary);
  text-align: center;
  word-break: break-word;
}

.drop-subtitle :deep(b) {
  color: var(--text-primary);
}

@media (max-width: 768px) {
  .drop-icon {
    font-size: 30px;
  }

  .drop-title {
    font-size: 15px;
  }

  .drop-subtitle {
    font-size: 12px;
  }
}
</style>
