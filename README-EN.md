# Wheat Embedding Toolkit

[简体中文](./README.md) | English

## Regenerate App Icons

After replacing `public/wheat_logo.png`, run:

```bash
yarn icons:regen
```

To use a different source image:

```bash
bash ./scripts/regenerate-icons.sh /path/to/logo.png
```

The default output uses a rounded-square background so the app icon looks fuller on macOS. To keep a transparent background:

```bash
ICON_STYLE=transparent yarn icons:regen
```

## General
![](images/en-1.png)

## Flash & Merge
* Support parsing firmware in file name: 
  Filename_Offset.bin: 'ESP32_0x222.bin'
  
* Support importing ESP-IDF project firmware into tool list

  your_porject/build/flasher_args.json

* Support importing the firmware of PlatformIO IDE project into the tool list

  your_porject/.pio/build/your_board/idedata.json

![](images/en-2.png)

## Partition Table
✅ Fill in the offset address of the partition table and align it

✅ Partition table firmware to CSV

![](images/en-3.png)


## Firmware
✅ History Path

✅ Search firmware

![](images/en-5.png)
