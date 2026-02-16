po_fa_15 = ["FA97", "FA98", "FA99", "FA00",
"FA01","FA02","FA03","FA04","FA05","FA06","FA07","FA08","FA09","FA10","FA11",
"FA12","FA13","FA14", "FA16", "FA17", "FA18", "FA19", "FA20", "FA21", "FA22","FA23", "FA24", "FA25", "FA26", "FA27", "FA28", "FA29", "FA30"]
po_ho_15 = ["HO96","HO98","HO99",
"HO00","HO02","HO03","HO04","HO05","HO06","HO07","HO08","HO09","HO10","HO11","HO12","HO13","HO14","HS72", "HO15","HO16",
		"HO17","HO18","HO19","HO20","HO21","HO22","HO23","HO24","HO25","HO26","HO27","HO28","HO29","HO30"]
po_su_15 = ["SU01","SU04","SU05","SU06","SU07","SU08","SU09","SU10","SU11","SU12","SU13","SU14", "SU15",
		"SU16",
		"SU17",
		"SU18",
		"SU19",
		"SU20",
		"SU21",
		"SU22",
		"SU23",
		"SU24",
		"SU25",
		"SU26",
		"SU27",
		"SU28",
		"SU29",
		"SU30"]
po_sp_15 = [
  "SP97","SP99","SP00","SP01","SP02","SP03","SP04","SP05","SP06","SP07","SP08","SP09","SP10","SP11","SP12","SP13","SP14",
  "SP15",
		"SP16",
		"SP17",
		"SP18",
		"SP19",
		"SP20",
		"SP21",
		"SP22",
		"SP23",
		"SP24",
		"SP25",
		"SP26",
		"SP27",
		"SP28",
		"SP29",
		"SP30"]

po_97 = ["SP97", "SU97", "HO97", "FA97"]
po_98 = ["SP98", "SU98", "HO98", "FA98"]
po_99 = ["SP99", "SU99", "HO99", "FA99"]
po_00 = ["SP00", "SU00", "HO00", "FA00"]
po_01 = ["SP01", "SU01", "HO01", "FA01"]
po_02 = ["SP02", "SU02", "HO02", "FA02"]
po_03 = ["SP03", "SU03", "HO03", "FA03"]
po_04 = ["SP04", "SU04", "HO04", "FA04"]
po_05 = ["SP05", "SU05", "HO05", "FA05"]
po_06 = ["SP06", "SU06", "HO06", "FA06"]
po_07 = ["SP07", "SU07", "HO07", "FA07"]
po_08 = ["SP08", "SU08", "HO08", "FA08"]
po_09 = ["SP09", "SU09", "HO09", "FA09"]
po_10 = ["SP10", "SU10", "HO10", "FA10"]
po_11 = ["SP11", "SU11", "HO11", "FA11"]
po_12 = ["SP12", "SU12", "HO12", "FA12"]
po_13 = ["SP13", "SU13", "HO13", "FA13"]
po_14 = ["SP14", "SU14", "HO14", "FA14"]

po_15 = ["SP15", "SU15", "HO15", "FA15"]
po_16 = ["SP16", "SU16", "HO16", "FA16"]
po_17 = ["SP17", "SU17", "HO17", "FA17"]
po_18 = ["SP18", "SU18", "HO18", "FA18"]
po_19 = ["SP19", "SU19", "HO19", "FA19"]
po_20 = ["SP20", "SU20", "HO20", "FA20"]
po_21 = ["SP21", "SU21", "HO21", "FA21"]
po_22 = ["SP22", "SU22", "HO22", "FA22"]
po_23 = ["SP23", "SU23", "HO23", "FA23"]
po_24 = ["SP24", "SU24", "HO24", "FA24"]
po_25 = ["SP25", "SU25", "HO25", "FA25"]
po_26 = ["SP26", "SU26", "HO26", "FA26"]
po_27 = ["SP27", "SU27", "HO27", "FA27"]
po_28 = ["SP28", "SU28", "HO28", "FA28"]
po_29 = ["SP29", "SU29", "HO29", "FA29"]
po_30 = ["SP30", "SU30", "HO30", "FA30"]
# # Print year-by-year results for each list
# for season, years in [("FA", po_fa_15), ("HO", po_ho_15), ("SU", po_su_15), ("SP", po_sp_15)]:
#     list = []
#     print(f"{season}:")
#     for year in years:
#         print(year)
#     print()
import base64
data = "eyJUeXBlIjoiTm90aWZpY2F0aW9uIiwiTWVzc2FnZUlkIjoiM2Y0Mzk2MzQtOTJjMi01MmFhLWJjZDQtZjc5NmFkMDRiMDAyIiwiVG9waWNBcm4iOiJhcm46YXdzOnNuczp1cy13ZXN0LTI6NzQ4NTk5Mzk2MTMwOmxwbS1mb3VuZGF0aW9uLXN0Zy1wcm9kdWN0LW1lcmNoLWRhdGEtY2hhbmdlLWV2ZW50cyIsIk1lc3NhZ2UiOiJ7XCJ0eXBlXCI6XCJhdHRyaWJ1dGlvbi5hc3NvY2lhdGlvbjp1cGRhdGVcIixcIm9iamVjdF9uYW1lXCI6XCJjcHNfc3R5bGVcIixcIm9iamVjdF9pZFwiOlwiSVIwMDU3XCIsXCJsb2NhdGlvblwiOlwiaHR0cHM6Ly9hdHRyaWJ1dGlvbi1zdGcubmlrZS1lMmUuY29tL3Byb2R1Y3RfbWVyY2gvYXNzb2NpYXRpb25zL2Nwc19zdHlsZS9JUjAwNTdcIixcImF0dHJpYnV0ZV9uYW1lXCI6XCJjY19zdWJfZGltZW5zaW9uXCIsXCJvbGRfdmFsdWVcIjpudWxsLFwibmV3X3ZhbHVlXCI6XCJDODc2NkVCMS1BNzJGLTExRUQtODBCRS0wMjQ3RDc3QkVFQTFcIixcInVwZGF0ZWRfYXRcIjpcIjIwMjUtMTEtMjdUMTI6MDE6MTAuODQ5WlwiLFwidXBkYXRlZF9ieVwiOlwibmlrZS5scG0ucHJvZHVjdC1tZXJjaFwiLFwiZGVyaXZhdGlvbl9qb2JfdXVpZFwiOlwiYjlmNzYwMTgtY2I4OC0xMWYwLThmYzEtMDY0OGNmYzZiMmFiXCIsXCJleHBsYW5hdGlvblwiOlwiQ3JlYXRpbmcgYSBkZXJpdmF0aW9uIGpvYiB3aXRoIGpvYl91dWlkIGI5Zjc2MDE4LWNiODgtMTFmMC04ZmMxLTA2NDhjZmM2YjJhYiAoZm9yIERlcml2YXRpb24gSm9iIElEIGI5Zjc2MDE4LWNiODgtMTFmMC04ZmMxLTA2NDhjZmM2YjJhYilcIn0iLCJUaW1lc3RhbXAiOiIyMDI1LTExLTI3VDEyOjAxOjEwLjg5MloiLCJTaWduYXR1cmVWZXJzaW9uIjoiMSIsIlNpZ25hdHVyZSI6IkM2ZUlTMjU0bzZGTmxRVkZPYnhZa1BWRlNpczJEM2p2U09FeklRZnZtL3JQSGhaM2U2YzE3YUJhaGtKaWpwRHlzZjVwbTNyMlgzMGpxUkk1Tk52WlB4NFZZY1FhdDhjVjFURHlVUkVrNGQ2SzltL2IwN1o5bkZaQm9jbTkwRVpUZ2d6TU1NL1VLeGwvb0I1L3JhUW1qVkZ0SHU2RTBuQVdyN2haNCtFeG9MVTNOQ3pEaVpvL3Z3aFVjMnlpZjNyanY3ZjlvcWNOdzZVV2FTWWNNWHZUYTRsaC9RaFRXTnZPOEpvdkQ4RHovVWdSS1I4SmsrTGFGamNmNHd1RkppZjhtYUV6eTlzUVU2RHBDN09ibEZFR3B0K0syL2Q4dHdIc3pCTHhlSytUTHZUVzJwcDNSb2FwTmdFWFUyaFhCZTkzRURzWDlIdVM1YjdaUU9JYi9qQkVZdz09IiwiU2lnbmluZ0NlcnRVUkwiOiJodHRwczovL3Nucy51cy13ZXN0LTIuYW1hem9uYXdzLmNvbS9TaW1wbGVOb3RpZmljYXRpb25TZXJ2aWNlLTYyMDljMTYxYzYyMjFmZGY1NmVjMWViNWM4MjFkMTEyLnBlbSIsIlVuc3Vic2NyaWJlVVJMIjoiaHR0cHM6Ly9zbnMudXMtd2VzdC0yLmFtYXpvbmF3cy5jb20vP0FjdGlvbj1VbnN1YnNjcmliZSZTdWJzY3JpcHRpb25Bcm49YXJuOmF3czpzbnM6dXMtd2VzdC0yOjc0ODU5OTM5NjEzMDpscG0tZm91bmRhdGlvbi1zdGctcHJvZHVjdC1tZXJjaC1kYXRhLWNoYW5nZS1ldmVudHM6ZTU2OGU0OTItNDljZS00ZGRiLTk0MDgtMmFmYWFiMzU5YmZkIiwiTWVzc2FnZUF0dHJpYnV0ZXMiOnsiZGVyaXZhdGlvbl9qb2JfdXVpZCI6eyJUeXBlIjoiU3RyaW5nIiwiVmFsdWUiOiJiOWY3NjAxOC1jYjg4LTExZjAtOGZjMS0wNjQ4Y2ZjNmIyYWIifSwib2JqZWN0X25hbWUiOnsiVHlwZSI6IlN0cmluZyIsIlZhbHVlIjoiY3BzX3N0eWxlIn0sImF0dHJpYnV0ZV9uYW1lIjp7IlR5cGUiOiJTdHJpbmciLCJWYWx1ZSI6ImNjX3N1Yl9kaW1lbnNpb24ifSwidHlwZSI6eyJUeXBlIjoiU3RyaW5nIiwiVmFsdWUiOiJhdHRyaWJ1dGlvbi5hc3NvY2lhdGlvbjp1cGRhdGUifSwicHJpb3JpdHkiOnsiVHlwZSI6IlN0cmluZyIsIlZhbHVlIjoibG93In0sIm9iamVjdF9pZCI6eyJUeXBlIjoiU3RyaW5nIiwiVmFsdWUiOiJJUjAwNTcifX19"
print(base64.b64decode(data).decode(
                    "utf-8"
                ))
