# GitHub Upload Guide - 15 Minutes

Quick step-by-step guide to upload your code to GitHub and get the repository link.

---

## ⏱️ Time Required: 15 minutes

- Create account: 5 minutes
- Create repository: 2 minutes  
- Upload files: 5 minutes
- Get link: 1 minute
- Send to supervisor: 2 minutes

---

## STEP 1: Create GitHub Account (5 minutes)

### If you already have a GitHub account, skip to Step 2!

1. Go to: **https://github.com/signup**

2. Enter your email address
   - Use your university email (e.g., username@uws.ac.uk)
   - Click "Continue"

3. Create a password
   - At least 8 characters
   - Mix of letters and numbers
   - Click "Continue"

4. Choose a username
   - Example: `sajid-cti-2024` or `msc-it-uws-2024`
   - Click "Continue"

5. Verify email
   - Check your email inbox
   - Enter the 6-digit code
   - Click "Verify"

6. Complete setup
   - Answer questions (optional - you can skip)
   - Click "Continue" / "Skip"

✅ **Account created!**

---

## STEP 2: Create New Repository (2 minutes)

1. Click the **green "New"** button (top left or top right)

2. Fill in repository details:
   ```
   Repository name: MSc-CTI-Robustness-Project
   
   Description: MSc IT Dissertation: Robust and Explainable Machine Learning 
                for Predictive Cyber Threat Intelligence
   
   Visibility: ✅ Public (so supervisor can see)
   
   ☑️ Add a README file (check this box)
   ```

3. Click **"Create repository"**

✅ **Repository created!**

---

## STEP 3: Upload Your Code Files (5 minutes)

### Method A: Web Upload (Easiest - No git knowledge needed)

1. In your new repository, click **"Add file"** → **"Upload files"**

2. **Drag and drop ALL these files** into the upload area:
   ```
   ✅ component1_baseline_models.py
   ✅ component2_adversarial_attacks.py
   ✅ component3_defense_mechanisms.py
   ✅ component4_explainability.py
   ✅ run_all.py
   ✅ requirements.txt
   ✅ README.md
   ✅ INSTALLATION_GUIDE.md
   ✅ Dataset_Phising_Website.csv
   
   Optional (if you already ran the code):
   ✅ results/ folder (with all CSV and PNG files)
   ✅ models/ folder (with all .pkl files)
   ```

3. Scroll down to **"Commit changes"**
   ```
   Commit message: "Initial commit - Full CTI robustness implementation"
   
   Description: "Complete implementation of 4-component framework including
                 baseline models, adversarial attacks, defenses, and explainability"
   ```

4. Click **"Commit changes"**

5. Wait for upload to complete (1-2 minutes)

✅ **Files uploaded!**

---

### Method B: Git Command Line (If you know git)

```bash
# In your project folder
git init
git add .
git commit -m "Initial commit - Full implementation"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/MSc-CTI-Robustness-Project.git
git push -u origin main
```

---

## STEP 4: Get Your Repository Link (1 minute)

1. In your repository, click the green **"Code"** button

2. Copy the HTTPS URL

**Your link will look like:**
```
https://github.com/YOUR-USERNAME/MSc-CTI-Robustness-Project
```

**Example:**
```
https://github.com/sajid-cti-2024/MSc-CTI-Robustness-Project
```

✅ **Link copied!**

---

## STEP 5: Send to Supervisor (2 minutes)

### Email Template:

```
Subject: MSc Dissertation - Code Repository Link

Dear Dr. Haider Ali,

Please find the implementation code repository for our MSc IT dissertation:

Repository: https://github.com/YOUR-USERNAME/MSc-CTI-Robustness-Project

The repository contains:
- All 4 component implementations (baseline models, attacks, defenses, explainability)
- Complete dataset (22,110 phishing website samples)
- Installation and execution guides
- Requirements file for reproducibility

All code is documented and ready to execute following the instructions in README.md.

Best regards,
Muhammad Sajid Iqbal
Aneela Shafique
Muhammad Arfan
Muhammad Talha Anwar

MSc Information Technology
University of the West of Scotland
```

**Send this email with your actual GitHub link!**

✅ **Done!**

---

## 📊 What Your Supervisor Will See

When your supervisor clicks the link, they'll see:

```
MSc-CTI-Robustness-Project
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 component1_baseline_models.py
📁 component2_adversarial_attacks.py  
📁 component3_defense_mechanisms.py
📁 component4_explainability.py
📁 run_all.py
📁 requirements.txt
📁 Dataset_Phising_Website.csv
📁 README.md
📁 INSTALLATION_GUIDE.md

README.md:
# MSc CTI Robustness and Explainability Framework

Robust and Explainable Machine Learning for Predictive 
Cyber Threat Intelligence

[Full professional documentation...]
```

✅ **Professional presentation!**

---

## 🎯 Repository Checklist

Make sure your repository has:

- [x] All 4 component Python files
- [x] run_all.py master script
- [x] requirements.txt
- [x] README.md
- [x] Dataset CSV file
- [x] INSTALLATION_GUIDE.md
- [x] Results folder (if you ran the code)
- [x] Models folder (if you ran the code)

---

## 💡 Pro Tips

### Tip 1: Make it look professional

Add a nice description at the top of your repository page:
- Click "About" (⚙️ gear icon on the right)
- Add: "MSc IT Dissertation - CTI Robustness Framework"
- Add topics: `machine-learning`, `cybersecurity`, `explainable-ai`
- Click "Save"

### Tip 2: Add results later

If you haven't run the code yet, you can:
1. Upload code files now
2. Run the code later
3. Upload results folder afterward

### Tip 3: Update if needed

You can always:
- Click "Add file" → "Upload files"
- Drag new/updated files
- They'll replace old versions

---

## 🛠️ Troubleshooting

### Problem: Upload fails

**Solution:** 
- GitHub has 100MB file limit
- If your files are larger, upload in smaller batches
- Or use Git LFS for large files

### Problem: Can't find "New" button

**Solution:**
- Look for green "New" button on left side
- Or click your profile icon → "Your repositories" → "New"

### Problem: Repository is Private

**Solution:**
- Go to repository Settings
- Scroll to "Danger Zone"
- Click "Change visibility" → "Make public"

---

## ✅ Success Indicators

You know it worked when:

1. ✅ You can open `https://github.com/YOUR-USERNAME/MSc-CTI-Robustness-Project`
2. ✅ You see all your files listed
3. ✅ Your supervisor can open the link (send them a test)
4. ✅ README.md displays properly formatted

---

## 🎉 You're Done!

**Total time: ~15 minutes**

Your code is now:
- ✅ Publicly accessible
- ✅ Professionally presented
- ✅ Ready for supervisor review
- ✅ Properly documented
- ✅ Reproducible by others

---

## 📞 Need Help?

**Common issues:**
- Forgot password? Click "Forgot password" on GitHub
- Upload too slow? Try smaller batches
- Wrong file uploaded? Just upload again (replaces old version)

**GitHub Help:**
- GitHub Docs: https://docs.github.com
- GitHub Support: https://support.github.com

---

**Next Step:** Wait for supervisor's response!

Good luck! 🚀
