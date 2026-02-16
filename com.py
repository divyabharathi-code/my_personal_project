["tertiaryColorDescription",
"tertiaryLabColorCode",
"secondaryColorAbbreviation",
"secondaryColorDescription",
"secondaryLabColorCode",
"logoColorDescription",
"logoColorAbbreviation",
"logoLabColorCode",
"primaryColorDescription",
"primaryColorAbbreviation"]

def compare_rows(r1, r2):
    fields1 = r1.split('\t')
    fields2 = r2.split('\t')
    diffs = []
    for i, (f1, f2) in enumerate(zip(fields1, fields2)):
        if f1 != f2:
            diffs.append(f"Column {i+1}:\n- {f1}\n+ {f2}")
    if not diffs:
        print("Rows are identical.")
    else:
        print("Differences found:")
        for diff in diffs:
            print(diff)

compare_rows(row1, row2)