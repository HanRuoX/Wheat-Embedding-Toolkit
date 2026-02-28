# Wheat Embedding Toolkit

简体中文 | [English](./README-EN.md)

## 重新生成桌面图标

替换 `public/wheat_logo.png` 后执行：

```bash
yarn icons:regen
```

如需指定其他源图：

```bash
bash ./scripts/regenerate-icons.sh /path/to/logo.png
```

默认会生成更适合 macOS 桌面的“圆角底板”图标。如果你想保留透明底：

```bash
ICON_STYLE=transparent yarn icons:regen
```

## 常规
![](images/zh-1.png)

## 烧录或合并固件
* 支持文件名中解析固件: 
  固件名称_烧录地址.bin: 'ESP32_0x222.bin'
  
* 支持导入ESP-IDF项目的固件到工具列表

  your_porject/build/flasher_args.json

* 支持导入PlatformIO IDE项目的固件到工具列表

  your_porject/.pio/build/your_board/idedata.json

![](images/zh-2.png)

## 分区表
✅ 填充分区表的偏移地址并对齐

✅ 分区表固件转CSV

![](images/zh-3.png)

## BLE
✅ 广播扫描

❎ 差广播过滤未完成

❎ 连接蓝牙

![](images/5.png)

## 固件管理
✅ 历史烧录路径

✅ 搜索固件
生成logo yarn tauri icon src/assets/wheat_logo.png
![](images/zh-5.png)
