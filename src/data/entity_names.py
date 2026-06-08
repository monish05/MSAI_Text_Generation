import csv
from pathlib import Path

def extract_names_from_csv(filepath):
    names = []
    try:
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return names
            name_columns = []
            for field in reader.fieldnames:
                field_lower = field.lower()
                if field_lower in ("name", "assignee name"):
                    name_columns = [field]

                    break
                if field_lower == "first name":
                    name_columns.append(field)

                elif field_lower == "last name":
                    name_columns.insert(0, field)
            for row in reader:
                if not name_columns:
                    continue
                if len(name_columns) == 1 and row.get(name_columns[0]):
                    name = row[name_columns[0]].strip()
                    if name and name.upper() != "NA":
                        names.append(name)

                elif len(name_columns) == 2:
                    last_name = (row.get(name_columns[0]) or "").strip()
                    first_name = (row.get(name_columns[1]) or "").strip()

                    if (
                        (last_name or first_name)
                        and last_name.upper() != "NA"
                        and (first_name.upper() != "NA")
                    ):
                        if last_name and first_name:
                            names.append(f"{first_name} {last_name}")
                        elif first_name:
                            names.append(first_name)

                        else:
                            names.append(last_name)
    except OSError:
        pass
    return names

def extract_names_from_text(filepath):
    names = []
    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line.startswith("Name:"):
                    name = line.replace("Name:", "", 1).strip()
                    if name:
                        names.append(name)
    except OSError:
        pass

    return names

def load_entity_names(archive_dir):
    all_names = set()
    if not archive_dir.exists():
        return []
    for filepath in sorted(archive_dir.iterdir()):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() == ".csv":
            all_names.update(extract_names_from_csv(filepath))

        elif filepath.suffix.lower() == ".txt":
            all_names.update(extract_names_from_text(filepath))
    return sorted(all_names)
