import sys

def refactor_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Ranges to delete (1-indexed, inclusive)
    # Note: We must delete from bottom to top to keep indices valid,
    # or just filter out the lines.
    
    ranges_to_delete = [
        (1549, 2118), # PMaker methods
        (3126, 3259), # Probing methods part 1
        (3754, 4232), # Probing methods part 2
    ]
    
    # Convert to 0-indexed set of lines to remove
    to_remove = set()
    for start, end in ranges_to_delete:
        for i in range(start - 1, end):
            to_remove.add(i)
            
    new_lines = []
    for i, line in enumerate(lines):
        if i not in to_remove:
            new_lines.append(line)
        elif i == 1548: # Replace the first block with a status method
            new_lines.append("    def _pmaker_status(self) -> dict:\n")
            new_lines.append("        return self.pmaker.status_dict()\n")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    refactor_file(sys.argv[1])
