import splitfolders

# Split the existing 'test/' folder into train and val sets
splitfolders.ratio(
    "data/fer2013/test",        # Input folder with class subfolders
    output="data/fer2013/",     # Output folder where train/ and val/ will be created
    seed=1337,
    ratio=(.8, .2),             # 80% train, 20% val
    move=True                   # Move files (set to False if you prefer to copy)
)

print("✅ Dataset split into train/ and val/")
