# 🚀 راهنمای اجرای آموزش در Background

## 📋 دستورات سریع:

### 1. شروع آموزش در background:

```bash
cd ~/paradiseModel
chmod +x scripts/*.sh
./scripts/run_training_background.sh
```

### 2. چک کردن وضعیت:

```bash
./scripts/check_training.sh
```

یا:

```bash
tail -f logs/training_*.log
```

### 3. تماشای log به صورت زنده:

```bash
./scripts/watch_training.sh
```

### 4. متوقف کردن آموزش:

```bash
./scripts/stop_training.sh
```

## 📝 دستورات دستی (بدون اسکریپت):

### شروع در background:

```bash
cd ~/paradiseModel
nohup python scripts/train_3080.py > logs/training.log 2>&1 &
echo $! > logs/training.pid
```

### چک کردن وضعیت:

```bash
# بررسی PID
cat logs/training.pid

# بررسی که آیا در حال اجرا است
ps -p $(cat logs/training.pid)

# دیدن آخرین خطوط log
tail -f logs/training.log
```

### متوقف کردن:

```bash
kill $(cat logs/training.pid)
```

## 🔍 دستورات مفید:

### دیدن استفاده از GPU:

```bash
watch -n 1 nvidia-smi
```

### دیدن آخرین خطوط log:

```bash
tail -n 50 logs/training_*.log
```

### جستجو در log:

```bash
grep "error" logs/training_*.log
grep "loss" logs/training_*.log
```

### بررسی فضای دیسک:

```bash
df -h
du -sh ~/paradiseModel
```

## 📊 مثال خروجی:

```bash
$ ./scripts/check_training.sh

==================================================================================
📊 وضعیت آموزش (Training Status)
==================================================================================

Started: Mon Jan 15 10:30:00 UTC 2024
PID: 12345
Log: /home/arisa/paradiseModel/logs/training_20240115_103000.log
Status: Running

✅ Training is running
📊 PID: 12345

==================================================================================
📝 آخرین خطوط Log (Last 20 lines):
==================================================================================
...
Epoch 2/3: 100%|████████| 722/722 [15:30<00:00, 1.23s/it, loss=0.45]
Evaluating: 100%|████████| 81/81 [00:45<00:00, 1.78it/s]
eval_loss: 0.48
...
```

## ⚠️ نکات مهم:

1. **Log Files**: تمام log ها در پوشه `logs/` ذخیره می‌شوند
2. **PID File**: PID در `logs/training.pid` ذخیره می‌شود
3. **Status File**: وضعیت در `logs/training_status.txt` ذخیره می‌شود
4. **Background**: آموزش حتی اگر terminal را ببندید ادامه می‌یابد

## 🎯 دستور یک خطی (پیشنهادی):

```bash
cd ~/paradiseModel && chmod +x scripts/*.sh && ./scripts/run_training_background.sh && sleep 2 && ./scripts/check_training.sh
```

این دستور:
1. به پوشه پروژه می‌رود
2. دسترسی اجرا می‌دهد
3. آموزش را در background شروع می‌کند
4. 2 ثانیه صبر می‌کند
5. وضعیت را نمایش می‌دهد

