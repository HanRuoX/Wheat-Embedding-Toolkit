<template>
  <a-config-provider
    :locale="locale"
    :theme="{
      algorithm: antThemeAlgorithm,
    }"
  >
    <Suspense>
      <RouterView />
    </Suspense>
  </a-config-provider>
</template>
<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, watch } from "vue";
import { theme } from "ant-design-vue";
import zhCN from "ant-design-vue/es/locale/zh_CN";
import enUS from "ant-design-vue/es/locale/en_US";
import i18n from "@/locales/i18n";
import { usePreferenceStore } from "@/stores/Preference";

const preferenceStore = usePreferenceStore();
if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
  preferenceStore.setSystemTheme(
    window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
  );
}

const locale = computed(() => (preferenceStore.language === "zh" ? zhCN : enUS));
const antThemeAlgorithm = computed(() =>
  preferenceStore.resolvedTheme === "light" ? theme.defaultAlgorithm : theme.darkAlgorithm
);

let stopSystemThemeListener: (() => void) | null = null;

watch(
  () => preferenceStore.language,
  (language) => {
    i18n.global.locale.value = language;
  },
  { immediate: true }
);

watch(
  () => preferenceStore.resolvedTheme,
  (themeMode) => {
    document.documentElement.setAttribute("data-theme", themeMode);
  },
  { immediate: true }
);

onMounted(() => {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return;
  }

  const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
  const handleSystemThemeChange = (event: MediaQueryListEvent) => {
    preferenceStore.setSystemTheme(event.matches ? "dark" : "light");
  };
  preferenceStore.setSystemTheme(mediaQuery.matches ? "dark" : "light");

  if (typeof mediaQuery.addEventListener === "function") {
    mediaQuery.addEventListener("change", handleSystemThemeChange);
    stopSystemThemeListener = () => mediaQuery.removeEventListener("change", handleSystemThemeChange);
    return;
  }

  mediaQuery.addListener(handleSystemThemeChange);
  stopSystemThemeListener = () => mediaQuery.removeListener(handleSystemThemeChange);
});

onBeforeUnmount(() => {
  stopSystemThemeListener?.();
  stopSystemThemeListener = null;
});
</script>
