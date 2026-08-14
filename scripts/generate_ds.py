import csv
import random

def generate_dataset(num_rows, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'age', 'height', 'weight'])
        
        for i in range(1, num_rows + 1):
            # 1. Generate Age (Matching your 11-62 range)
            age = random.randint(11, 65)
            
            # 2. Generate Height
            if age < 18:
                # Younger teens are generally shorter 
                height = int(random.gauss(150 + (age - 11) * 4, 8))
            else:
                # Adults are normally distributed around 172cm
                height = int(random.gauss(172, 10))
            
            # Constrain height to realistic bounds from your sample (140cm - 195cm)
            height = max(140, min(195, height))
            
            # 3. Generate Weight based on realistic BMI (18.5 - 29.9)
            bmi = random.uniform(18.5, 29.9)
            weight = int(bmi * ((height / 100) ** 2))
            
            writer.writerow([i, age, height, weight])

generate_dataset(1000, './data/data.csv')
print("Successfully generated data.csv with 1000 rows.")