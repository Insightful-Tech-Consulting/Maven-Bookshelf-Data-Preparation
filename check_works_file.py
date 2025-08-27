with open("Data-Works.csv", encoding="utf-8") as f:
    for i, line in enumerate(f):
        fields = line.strip().split(",")
        if len(fields) != 21:
            print(f"Row {i + 1} has {len(fields)} columns")
