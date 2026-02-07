<template>
  <a-row>
    <a-col :span="16">
      <div style="height: calc(100vh - 160px); overflow: auto">
        <a-list
          size="small"
          item-layout="vertical"
          :data-source="filteredData"
          :locale="{ emptyText: undefined }"
        >
          <template #header>
            <a-badge
              :count="filteredData.length"
              :number-style="{ backgroundColor: '#1890ff' }"
              :offset="[10, 0]"
              show-zero
            >
              <span style="color: rgba(255, 255, 255, 0.65)">
                {{ $t('ble.deviceCount', { count: filteredData.length }) }}
              </span>
            </a-badge>
          </template>
          <template #renderItem="{ item }">
            <a-list-item>
              <template #actions>
                <span :style="{ color: rssiColor(item.rssi) }">
                  <WifiOutlined style="margin-right: 4px" />
                  {{ item.rssi }} dBm
                </span>
              </template>
              <template #extra>
                <div style="width: 80px">
                  <a-progress
                    :percent="Math.max(0, 100 - (now - item.time) * 10)"
                    :size="10"
                    :showInfo="false"
                  />
                </div>
              </template>
              <a-list-item-meta>
                <template #title>
                  <p v-if="item.local_name" v-copy>{{ item.local_name }}</p>
                  <p v-else style="color: rgba(255, 255, 255, 0.35); font-style: italic">
                    {{ $t('ble.unknownDevice') }}
                  </p>
                </template>
                <template #description>
                  <p v-copy>{{ item.address }}</p>

                  <template v-if="item.services.length != 0">
                    <a-tag
                      style="margin-bottom: 3px"
                      color="blue"
                      v-for="service in item.services"
                      :key="service"
                      v-copy
                      >{{ service }}</a-tag
                    >
                  </template>

                  <a-tag
                    style="margin-bottom: 3px; white-space: normal; word-break: break-all"
                    v-if="item.adv.length != 0"
                    color="cyan"
                    v-copy
                    >{{
                      item.adv
                        .map((x: number) => x.toString(16).padStart(2, "0"))
                        .join(" ")
                        .toUpperCase()
                    }}</a-tag
                  >
                  <template v-if="Object.keys(item.manufacturer_data).length != 0">
                    <a-tag
                      style="margin-bottom: 3px; white-space: normal; word-break: break-all"
                      v-for="key in Object.keys(item.manufacturer_data)"
                      :key="key"
                      color="cyan"
                      v-copy
                      >[0x{{ Number(key).toString(16).toUpperCase().padStart(4, "0") }}]
                      {{
                        item.manufacturer_data[key]
                          .map((x: number) => x.toString(16).padStart(2, "0"))
                          .join(" ")
                          .toUpperCase()
                      }}</a-tag
                    >
                  </template>
                </template>
              </a-list-item-meta>
            </a-list-item>
          </template>
          <template #empty>
            <a-empty
              :description="scanState ? $t('ble.filter') + '...' : $t('ble.startScanning')"
            />
          </template>
        </a-list>
      </div></a-col
    >
    <a-col :span="8">
      <div style="margin: 5px">
        <a-button
          style="margin-bottom: 5px"
          type="primary"
          @click="scan"
          :danger="scanState"
          block
          >{{ scanBtnText }}</a-button
        >
        <a-card size="small" :title="$t('ble.filter')" style="margin-bottom: 5px">
          <a-input
            style="margin-bottom: 5px"
            v-model:value="filter.name"
            :addon-before="$t('ble.name')"
            allowClear
          />
          <a-input
            style="margin-bottom: 5px"
            v-model:value="filter.address"
            addon-before="MAC"
            allowClear
          />
          <a-input
            style="margin-bottom: 5px"
            v-model:value="filter.adv"
            :addon-before="$t('ble.advertising')"
            allowClear
          />
          <a-input
            style="margin-bottom: 5px"
            v-model:value="filter.uuid"
            addon-before="UUID"
            allowClear
          />
          <a-input-number
            prefix="-"
            addon-before="RSSI"
            :min="1"
            :max="100"
            style="width: 100%"
            v-model:value="filter.rssi"
          />
          <a-slider
            style="margin-bottom: 5px"
            v-model:value="filter.rssi"
            :min="0"
            :max="100"
            tooltipPlacement="bottom"
            :tipFormatter="
              (x:number) => {
                return '-' + x;
              }
            "
          />

          <a-button type="primary" @click="reset" block>{{
            i18n.global.t("ble.reset")
          }}</a-button>
        </a-card>
      </div>
    </a-col>
  </a-row>
</template>
<script setup lang="ts">
import { reactive, ref, computed } from "vue";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import { WifiOutlined } from "@ant-design/icons-vue";
import { message } from "ant-design-vue";
import moment from "moment";
import i18n from "@/locales/i18n";
const appWindow = getCurrentWebviewWindow()

const allData = ref([] as any);
const scanBtnText = ref(i18n.global.t("ble.startScanning"));
const scanState = ref(false);
const now = ref(moment().unix());
const filter = reactive({
  name: "",
  address: "",
  adv: "",
  uuid: "",
  rssi: 100,
});

let timer = {} as NodeJS.Timer;

const rssiColor = (rssi: number): string => {
  if (rssi >= -50) return "#52c41a";
  if (rssi >= -70) return "#faad14";
  return "#f5222d";
};

const filteredData = computed(() => {
  let result = allData.value;

  if (filter.name) {
    result = result.filter((x: any) =>
      x.local_name.toLowerCase().includes(filter.name.toLowerCase())
    );
  }

  if (filter.address) {
    result = result.filter((x: any) =>
      x.address.toLowerCase().includes(filter.address.toLowerCase())
    );
  }

  if (filter.uuid) {
    result = result.filter(
      (x: any) =>
        x.services.filter((s: string) => s.includes(filter.uuid)).length > 0
    );
  }

  if (filter.adv) {
    const advFilter = filter.adv.toLowerCase().replace(/\s/g, "");
    result = result.filter((x: any) => {
      // Check adv data
      const advHex = x.adv
        .map((b: number) => b.toString(16).padStart(2, "0"))
        .join("")
        .toLowerCase();
      if (advHex.includes(advFilter)) return true;

      // Check manufacturer_data
      for (const key of Object.keys(x.manufacturer_data)) {
        const mfHex = x.manufacturer_data[key]
          .map((b: number) => b.toString(16).padStart(2, "0"))
          .join("")
          .toLowerCase();
        if (mfHex.includes(advFilter)) return true;
      }
      return false;
    });
  }

  if (filter.rssi < 100) {
    result = result.filter((x: any) => x.rssi >= -filter.rssi);
  }

  // Sort by RSSI descending (strongest first)
  return [...result].sort((a: any, b: any) => b.rssi - a.rssi);
});

await listen("ble_advertisement_scan_event", (event: any) => {
  let peripheral = JSON.parse(event.payload);

  let ble = allData.value.find((x: any) => x.address == peripheral.address);
  if (ble == undefined) {
    peripheral.time = moment().unix();
    allData.value.push(peripheral);
  } else {
    allData.value.map((item: any) => {
      if (item.address === peripheral.address) {
        item.rssi = peripheral.rssi;
        item.manufacturer_data = peripheral.manufacturer_data;
        item.time = moment().unix();
      }
    });
  }
});

const reset = () => {
  filter.name = "";
  filter.address = "";
  filter.adv = "";
  filter.uuid = "";
  filter.rssi = 100;
};

const scan = async () => {
  if (scanState.value) {
    scanBtnText.value = i18n.global.t("ble.startScanning");
    scanState.value = false;
    appWindow.emit("stop_ble_advertisement_scan", {});
    clearInterval(timer);
  } else {
    allData.value = [];
    scanBtnText.value = i18n.global.t("ble.stopScanning");
    scanState.value = true;
    timer = setInterval(() => {
      now.value = moment().unix();
      allData.value = allData.value.filter(
        (x: any) => moment().unix() - x.time <= 10
      );
    }, 1000);
    try {
      await invoke("start_ble_advertisement_scan", {});
    } catch (error) {
      scanState.value = false;
      scanBtnText.value = i18n.global.t("ble.startScanning");
      clearInterval(timer);
      message.error(String(error));
    }
  }
};
</script>
