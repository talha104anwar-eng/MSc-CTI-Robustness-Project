# Installation Guide

Complete step-by-step installation guide for the MSc CTI Robustness Project.

---

## ⏱️ Time Required

- **Installation:** 15-20 minutes
- **First run:** 25-30 minutes
- **Total:** ~45 minutes

---

## 📋 Prerequisites

Before starting, ensure you have:
- [ ] A computer with Windows, Mac, or Linux
- [ ] Internet connection (for downloading Python and libraries)
- [ ] At least 4GB RAM (8GB recommended)
- [ ] At least 2GB free disk space

---

## STEP 1: Install Python

### Windows:

1. Go to: https://www.python.org/downloads/release/python-397/
2. Download: `Windows installer (64-bit)`
3. Run the installer
4. ⚠️ **CRITICAL:** Check ☑️ "Add Python 3.9 to PATH"
5. Click "Install Now"
6. Wait for completion
7. Click "Close"

### Mac:

1. Go to: https://www.python.org/downloads/release/python-397/
2. Download: `macOS 64-bit universal2 installer`
3. Run the `.pkg` file
4. Follow installer (Continue → Install)
5. Enter password when prompted
6. Wait for completion

### Linux (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install python3.9 python3-pip
```

---

## STEP 2: Verify Python Installation

Open Terminal/Command Prompt and run:

```bash
python --version
```

**Expected output:**
```
Python 3.9.7
```

If you see this, ✅ Python is installed correctly!

If you see "python is not recognized", see Troubleshooting below.

---

## STEP 3: Download Project Files

### Option A: From GitHub (if available)

```bash
git clone https://github.com/yourusername/MSc-CTI-Robustness-Project.git
cd MSc-CTI-Robustness-Project
```

### Option B: From ZIP file

1. Extract the ZIP file
2. Open terminal in the extracted folder
3. Verify you see these files:
   - component1_baseline_models.py
   - component2_adversarial_attacks.py
   - component3_defense_mechanisms.py
   - component4_explainability.py
   - requirements.txt
   - Dataset_Phising_Website.csv

---

## STEP 4: Install Required Libraries

In the project folder, run:

```bash
pip install -r requirements.txt
```

**What this does:** Installs all necessary Python libraries (~800MB download)

**Expected time:** 5-10 minutes

You'll see output like:
```
Collecting pandas==1.3.5
  Downloading pandas-1.3.5...
Installing collected packages: pandas, numpy, scikit-learn...
Successfully installed pandas-1.3.5 numpy-1.21.6 ...
```

---

## STEP 5: Verify Installation

Run this test command:

```bash
python -c "import pandas, numpy, sklearn, imblearn, art, shap, lime, matplotlib, seaborn; print('✅ All libraries installed successfully!')"
```

**Expected output:**
```
✅ All libraries installed successfully!
```

If you see this, you're ready to run the code! 🎉

---

## 🎯 Quick Test Run

To verify everything works, run:

```bash
python component1_baseline_models.py
```

This will:
1. Load the dataset
2. Train 3 models
3. Generate results
4. Create visualizations

**Expected time:** 3-5 minutes

If this completes without errors, your installation is complete! ✅

---

## 🛠️ Troubleshooting

### Problem: "python is not recognized"

**Windows Solution:**
1. Search "Environment Variables" in Start Menu
2. Click "Edit the system environment variables"
3. Click "Environment Variables" button
4. Under "System Variables", find "Path"
5. Click "Edit"
6. Click "New"
7. Add: `C:\Python39\`
8. Click "New" again
9. Add: `C:\Python39\Scripts\`
10. Click OK on all windows
11. Close and reopen terminal
12. Try `python --version` again

**Mac Solution:**
```bash
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

---

### Problem: "No module named 'sklearn'"

**Solution:**
```bash
pip install scikit-learn
```

---

### Problem: "Permission denied"

**Windows Solution:**
- Right-click Command Prompt
- Choose "Run as Administrator"
- Try installation again

**Mac/Linux Solution:**
```bash
pip install --user -r requirements.txt
```

---

### Problem: "Microsoft Visual C++ required" (Windows)

**Solution:**
1. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Install it
3. Try `pip install -r requirements.txt` again

---

### Problem: Installation is very slow

**Solution:**
This is normal for `adversarial-robustness-toolbox` (large library ~500MB)

Just wait 5-10 minutes. You'll see:
```
Downloading adversarial-robustness-toolbox-1.10.1...
[████████████████████████████████] 100%
```

---

## ✅ Installation Complete!

If all steps completed successfully, you should have:

- [x] Python 3.9.7 installed
- [x] All 9 libraries installed
- [x] Project files in a folder
- [x] Dataset file present
- [x] Test run successful

---

## 🚀 Next Steps

1. **To run all components:**
   ```bash
   python run_all.py
   ```

2. **To run individual components:**
   ```bash
   python component1_baseline_models.py
   python component2_adversarial_attacks.py
   python component3_defense_mechanisms.py
   python component4_explainability.py
   ```

3. **To view results:**
   - Check `results/` folder for CSV files
   - Check `results/` folder for PNG visualizations
   - Check `models/` folder for trained models

---

## 💡 Tips

- Close other programs before running (frees up RAM)
- Let each component finish completely
- Don't close terminal while running
- Results will be in `results/` folder
- Models will be in `models/` folder

---

## 📞 Need Help?

If you encounter issues not covered here:
1. Check the error message carefully
2. Google the specific error
3. Check Python library documentation
4. Ask supervisor for assistance

---

**Installation time: 15-20 minutes**  
**Ready to run!** 🎉
