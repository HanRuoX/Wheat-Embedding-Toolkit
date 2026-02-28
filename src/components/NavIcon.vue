<template>
  <span class="nav-icon" :style="iconStyle">
    <img v-if="isImageIcon" :src="icon" :alt="alt" class="nav-icon-image" />
    <span v-else class="nav-icon-text">{{ icon }}</span>
  </span>
</template>

<script setup lang="ts">
import { computed } from "vue";

const props = withDefaults(
  defineProps<{
    icon?: string;
    alt?: string;
    size?: string;
  }>(),
  {
    icon: "",
    alt: "",
    size: "1em",
  }
);

const isImageIcon = computed(() => {
  const value = props.icon;
  return (
    value.startsWith("data:image/") ||
    /\.(svg|png|jpe?g|webp|ico)(\?.*)?$/i.test(value)
  );
});

const iconStyle = computed(() => ({
  fontSize: props.size,
}));
</script>

<style scoped>
.nav-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  line-height: 1;
}

.nav-icon-image {
  width: 1em;
  height: 1em;
  display: block;
  object-fit: contain;
}

.nav-icon-text {
  line-height: 1;
}
</style>
