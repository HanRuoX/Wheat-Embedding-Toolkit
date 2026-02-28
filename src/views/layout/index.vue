<template>
  <div
    class="wechat-shell"
    :class="{ 'menu-collapsed': isMenuCollapsed, resizing: isResizing }"
    :style="shellStyle"
  >
    <aside class="dock-rail">
      <div class="dock-main">
        <img class="dock-logo" :src="wheatLogo" alt="Wheat Embedding Toolkit" />
        <template v-if="isMenuCollapsed">
          <a-popover
            v-for="section in dockSections"
            :key="section.key"
            placement="rightTop"
            trigger="hover"
          >
            <template #content>
              <div class="dock-popover">
                <div class="dock-popover-title">{{ section.title }}</div>
                <button
                  v-for="menu in section.menus"
                  :key="String(menu.name)"
                  class="dock-popover-item"
                  :class="{ active: isMenuActive(menu) }"
                  type="button"
                  @click="to(menu)"
                >
                  <NavIcon :icon="String(menu.meta?.icon || '')" :alt="String(menu.meta?.title || '')" />
                  <span>{{ menu.meta?.title }}</span>
                </button>
              </div>
            </template>
            <button
              class="dock-item"
              :class="{ active: isActiveSection(section) }"
              type="button"
              @click="onDockSectionClick(section)"
            >
              <NavIcon class="dock-icon" :icon="section.icon" :alt="section.title" size="20px" />
            </button>
          </a-popover>
        </template>
        <button
          v-else
          v-for="section in dockSections"
          :key="section.key"
          class="dock-item"
          :class="{ active: isActiveSection(section) }"
          type="button"
          @click="switchSection(section)"
        >
          <NavIcon class="dock-icon" :icon="section.icon" :alt="section.title" size="20px" />
        </button>
      </div>
      <div class="dock-secondary">
        <a-tooltip :title="$t('setting.multiWindow')" placement="right">
          <button
            class="dock-item"
            type="button"
            @click="openNewWindow"
          >
            <CopyOutlined class="dock-icon dock-ant-icon" />
          </button>
        </a-tooltip>
        <a-tooltip v-if="settingMenu" :title="String(settingMenu.meta?.title || $t('menu.setting'))" placement="right">
          <button
            class="dock-item"
            :class="{ active: route.name === settingMenu.name }"
            type="button"
            @click="to(settingMenu)"
          >
            <NavIcon
              class="dock-icon"
              :icon="String(settingMenu.meta?.icon || '⚙️')"
              :alt="String(settingMenu.meta?.title || $t('menu.setting'))"
              size="20px"
            />
          </button>
        </a-tooltip>
      </div>
      <button
        class="dock-mid-toggle"
        type="button"
        @click="toggleMenuPanel"
      >
        {{ isMenuCollapsed ? "›" : "‹" }}
      </button>
    </aside>

    <aside class="menu-panel" :class="{ collapsed: isMenuCollapsed }">
      <div class="menu-panel-title">{{ activeSection?.title || "工具" }}</div>
      <a-menu class="menu-list" :selected-keys="menuSelectedKeys" mode="inline">
        <a-menu-item
          v-for="menu in panelMenus"
          :key="String(menu.name)"
          @click="to(menu)"
        >
          <NavIcon :icon="String(menu.meta?.icon || '')" :alt="String(menu.meta?.title || '')" />
          <span>{{ menu.meta?.title }}</span>
        </a-menu-item>
      </a-menu>
      <div
        v-if="!isMenuCollapsed"
        class="menu-resize-handle"
        @mousedown="startResize"
      />
    </aside>
    <main class="main-panel">
      <header class="main-header">
        <h2 class="main-title">{{ route.meta?.title || "Wheat Embedding Toolkit" }}</h2>
        <div id="main-header-actions" class="main-header-actions" />
      </header>
      <section class="main-content">
        <router-view />
      </section>
      <div class="main-terminal" v-if="showGlobalTerminal">
        <Terminal />
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import i18n from "@/locales/i18n";
import esp32Icon from "@/assets/esp32-icon.png";
import wheatLogo from "@/assets/wheat_logo.png";
import Terminal from "@/components/Terminal.vue";
import NavIcon from "@/components/NavIcon.vue";
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { useRoute, useRouter, RouteRecordRaw } from "vue-router";
import { storeToRefs } from "pinia";
import { CopyOutlined } from "@ant-design/icons-vue";
import { message } from "ant-design-vue";
import { useToolsStore } from "@/stores/Tool";
import { spawnNewInstance } from "@/utils/common";

interface DockSection {
  key: string;
  icon: string;
  title: string;
  menus: RouteRecordRaw[];
}

const store = useToolsStore();
const { selectedKeys } = storeToRefs(store);

const DOCK_WIDTH = 60;
const MENU_WIDTH_MIN = 170;
const MENU_WIDTH_MAX = 320;
const MENU_WIDTH_DEFAULT = 200;
const MENU_COLLAPSED_STORAGE_KEY = "menuPanelCollapsed";

const router = useRouter();
const route = useRoute();

const rootChildren = ref((router.options.routes[0].children || []) as RouteRecordRaw[]);
const toolsGroup = computed(
  () =>
    (rootChildren.value.find((item) => item.name === "tools")?.children || []) as RouteRecordRaw[]
);
const settingMenu = computed(() =>
  rootChildren.value.find((item) => item.name === "setting")
);

const findToolRoute = (name: string) =>
  toolsGroup.value.find((item) => item.name === name && item.meta?.display);

const dockSections = computed<DockSection[]>(() => {
  const serial = findToolRoute("serial");
  const ble = findToolRoute("ble");
  const esp32Menus = ["basic", "flash", "partition", "firmware"]
    .map((name) => findToolRoute(name))
    .filter((item): item is RouteRecordRaw => !!item);

  const sections: DockSection[] = [];

  if (serial) {
    sections.push({
      key: "serial",
      icon: String(serial.meta?.icon || "🧵"),
      title: String(serial.meta?.title || i18n.global.t("menu.serialAssistant")),
      menus: [serial],
    });
  }

  if (esp32Menus.length > 0) {
    sections.push({
      key: "esp32",
      icon: esp32Icon,
      title: String(i18n.global.t("menu.esp32Assistant")),
      menus: esp32Menus,
    });
  }

  if (ble) {
    sections.push({
      key: "ble",
      icon: String(ble.meta?.icon || "🐳"),
      title: String(ble.meta?.title || "BLE"),
      menus: [ble],
    });
  }

  return sections;
});

const activeSection = computed(() => {
  const currentName = route.name;
  if (settingMenu.value?.name === currentName) {
    return {
      key: "setting",
      icon: String(settingMenu.value.meta?.icon || "⚙️"),
      title: String(settingMenu.value.meta?.title || i18n.global.t("menu.setting")),
      menus: [settingMenu.value],
    } as DockSection;
  }
  return (
    dockSections.value.find((section) =>
      section.menus.some((menu) => menu.name === currentName)
    ) || dockSections.value[0]
  );
});

const panelMenus = computed(() => activeSection.value?.menus || []);

const menuSelectedKeys = computed(() => selectedKeys.value.map((item) => String(item ?? "")));

watch(
  () => route.name,
  (name) => {
    selectedKeys.value = [name];
  },
  { immediate: true }
);

const showGlobalTerminal = computed(
  () =>
    route.name !== "serial" &&
    route.name !== "setting" &&
    route.name !== "ble"
);
const initialMenuCollapsed = (() => {
  const saved = localStorage.getItem(MENU_COLLAPSED_STORAGE_KEY);
  if (saved === null) {
    return true;
  }
  return saved === "1";
})();
const isMenuCollapsed = ref(initialMenuCollapsed);
const isResizing = ref(false);
const initialMenuWidth = (() => {
  const saved = Number(localStorage.getItem("menuPanelWidth"));
  if (Number.isFinite(saved) && saved >= MENU_WIDTH_MIN && saved <= MENU_WIDTH_MAX) {
    return saved;
  }
  return MENU_WIDTH_DEFAULT;
})();
const menuWidth = ref(initialMenuWidth);

const shellStyle = computed(() => ({
  "--dock-width": `${DOCK_WIDTH}px`,
  "--menu-width": isMenuCollapsed.value ? "0px" : `${menuWidth.value}px`,
}));

const isActiveSection = (section: DockSection) => {
  return activeSection.value?.key === section.key;
};

const isMenuActive = (menu: RouteRecordRaw) => {
  return route.name === menu.name;
};

const onDockSectionClick = (section: DockSection) => {
  if (isMenuCollapsed.value && section.menus.length > 1) {
    return;
  }
  switchSection(section);
};

const switchSection = (section: DockSection) => {
  const first = section.menus[0];
  if (first) {
    to(first);
  }
};

const to = (data: RouteRecordRaw) => {
  router.push(data.path);
};

const openNewWindow = async () => {
  try {
    await spawnNewInstance();
  } catch (error) {
    message.error(
      error instanceof Error ? error.message : i18n.global.t("setting.multiWindow")
    );
  }
};

const toggleMenuPanel = () => {
  isMenuCollapsed.value = !isMenuCollapsed.value;
  localStorage.setItem(MENU_COLLAPSED_STORAGE_KEY, isMenuCollapsed.value ? "1" : "0");
};

const clampWidth = (width: number) =>
  Math.max(MENU_WIDTH_MIN, Math.min(MENU_WIDTH_MAX, width));

let resizeStartX = 0;
let resizeStartWidth = initialMenuWidth;

const onResizeMove = (event: MouseEvent) => {
  if (!isResizing.value) {
    return;
  }
  const delta = event.clientX - resizeStartX;
  menuWidth.value = clampWidth(resizeStartWidth + delta);
};

const stopResize = () => {
  if (!isResizing.value) {
    return;
  }
  isResizing.value = false;
  document.body.style.cursor = "";
  document.body.style.userSelect = "";
  localStorage.setItem("menuPanelWidth", String(menuWidth.value));
  window.removeEventListener("mousemove", onResizeMove);
  window.removeEventListener("mouseup", stopResize);
};

const startResize = (event: MouseEvent) => {
  if (isMenuCollapsed.value) {
    return;
  }
  event.preventDefault();
  isResizing.value = true;
  resizeStartX = event.clientX;
  resizeStartWidth = menuWidth.value;
  document.body.style.cursor = "col-resize";
  document.body.style.userSelect = "none";
  window.addEventListener("mousemove", onResizeMove);
  window.addEventListener("mouseup", stopResize);
};

onBeforeUnmount(() => {
  stopResize();
});
</script>

<style scoped>
.wechat-shell {
  display: grid;
  grid-template-columns: var(--dock-width) var(--menu-width) minmax(0, 1fr);
  height: 100vh;
  min-height: 100vh;
  background: var(--panel-bg);
  color: var(--text-secondary);
  overflow: hidden;
  transition: grid-template-columns 0.2s ease;
}

.wechat-shell.menu-collapsed {
  grid-template-columns: var(--dock-width) 0 minmax(0, 1fr);
}

.wechat-shell.resizing {
  transition: none;
}

.dock-rail {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
  min-height: 0;
  padding: 14px 10px;
  background: var(--panel-bg-strong);
  border-right: 1px solid var(--panel-border);
}

.dock-main {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.dock-secondary {
  margin-top: auto;
  margin-bottom: 22px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.dock-logo {
  width: 40px;
  height: 40px;
  display: block;
  object-fit: contain;
  margin-bottom: 8px;
}

.dock-item {
  width: 36px;
  height: 36px;
  border: 0;
  border-radius: 10px;
  background: transparent;
  color: var(--text-secondary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.dock-item:hover {
  background: var(--panel-border);
}

.dock-item.active {
  background: var(--accent-soft);
  color: var(--text-primary);
}

.dock-icon {
  font-size: 16px;
  line-height: 1;
}

.dock-ant-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.dock-mid-toggle {
  position: absolute;
  top: 50%;
  right: -12px;
  transform: translateY(-50%);
  width: 24px;
  height: 30px;
  border: 1px solid var(--panel-border);
  border-radius: 8px;
  background: var(--panel-bg);
  color: var(--text-primary);
  cursor: pointer;
  font-size: 13px;
  line-height: 1;
  padding: 0;
  z-index: 3;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.2s ease, border-color 0.2s ease;
}

.dock-mid-toggle:hover {
  background: var(--accent-soft);
  border-color: var(--accent);
}

.dock-popover {
  min-width: 170px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.dock-popover-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  padding-bottom: 4px;
  border-bottom: 1px solid var(--panel-border);
}

.dock-popover-item {
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--text-secondary);
  height: 32px;
  width: 100%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 8px;
  padding: 0 10px;
}

.dock-popover-item:hover {
  background: var(--panel-border);
}

.dock-popover-item.active {
  background: var(--accent-soft);
  color: var(--text-primary);
}

.menu-panel {
  display: flex;
  flex-direction: column;
  position: relative;
  height: 100%;
  min-height: 0;
  min-width: 0;
  border-right: 1px solid var(--panel-border);
  background: var(--panel-bg);
  overflow: hidden;
  transition: border-color 0.2s ease, opacity 0.2s ease;
}

.menu-panel.collapsed {
  border-right-color: transparent;
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
}

.menu-resize-handle {
  position: absolute;
  top: 0;
  right: -4px;
  width: 8px;
  height: 100%;
  cursor: col-resize;
  background: transparent;
  z-index: 2;
}

.menu-panel-title {
  padding: 18px 16px;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  border-bottom: 1px solid var(--panel-border);
}

.menu-list {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  border-right: 0;
  background: transparent;
}

.menu-list :deep(.ant-menu) {
  overflow-x: hidden;
}

.menu-list :deep(.ant-menu-item) {
  margin-inline: 8px;
  border-radius: 8px;
}

.menu-list :deep(.ant-menu-item-selected) {
  background: var(--accent-soft);
  color: var(--text-primary);
}

.main-panel {
  height: 100%;
  min-height: 0;
  min-width: 0;
  display: flex;
  flex-direction: column;
  background: var(--app-bg);
  overflow: hidden;
}

.main-header {
  height: 64px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 20px;
  border-bottom: 1px solid var(--panel-border);
  background: var(--panel-bg-strong);
}

.main-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
}

.main-header-actions {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.main-content {
  flex: 1;
  min-height: 0;
  overflow: auto;
}

.main-terminal {
  border-top: 1px solid var(--panel-border);
}
</style>
