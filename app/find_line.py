def find_string(filename, search_term):
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if search_term in line:
                print(f"Line {i+1}: {line.strip()}")

if __name__ == "__main__":
    find_string("debug.html", "レーン")
