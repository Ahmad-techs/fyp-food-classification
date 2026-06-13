
# Food-11 Classes (11 total)
CLASS_NAMES = [
    'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 
    'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit'
]

# Mapping: 11 Classes -> 4 Dietary Groups
# 0: Carbohydrate, 1: Protein, 2: Dessert, 3: Fiber/Other
COARSE_MAPPING = {
    'Bread': 0, 'Noodles-Pasta': 0, 'Rice': 0,
    'Meat': 1, 'Egg': 1, 'Seafood': 1, 'Dairy product': 1,
    'Dessert': 2,
    'Fried food': 3, 'Soup': 3, 'Vegetable-Fruit': 3
}

def get_coarse_label(fine_label_name):
    return COARSE_MAPPING.get(fine_label_name, 3) # Default to 'Other'

# Verify the mapping
if __name__ == "__main__":
    for cls in CLASS_NAMES:
        print(f"{cls} -> {get_coarse_label(cls)}")