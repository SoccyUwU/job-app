import pandas
import re
from collections import defaultdict
import json

class techJobClassifier:
    def __init__(self):
        # define all the categories
        with open("keywords.json") as file:
            self.keyword_patterns = json.load(file)

    def extract_technologies(self, text):
        results = defaultdict(list)

        text_lower = text.lower()

        for category, patterns in self.keyword_patterns.items():
            for tech, pattern in patterns.items():
                if re.search(pattern, text_lower):
                    results[category].append(tech)

        return dict(results)

    def process_batch(self, job_postings, output_file='tech_classifications.csv'):
        """Process a batch of job postings and save results to CSV."""
        results = []
        
        for idx, posting in enumerate(job_postings):
            classification = self.extract_technologies(posting)
            
            # Flatten the results for CSV format
            flat_result = {
                'posting_id': idx,
                'programming_languages': ', '.join(classification.get('programming_languages', [])),
                'frameworks_libraries': ', '.join(classification.get('frameworks_libraries', [])),
                'databases': ', '.join(classification.get('databases', [])),
                'cloud_platforms': ', '.join(classification.get('cloud_platforms', [])),
                'tools': ', '.join(classification.get('tools', [])),
            }
            results.append(flat_result)
        
        # Convert to DataFrame and save
        df = pandas.DataFrame(results)
        df.to_csv(output_file, index=False)
        return df

# Example usage
if __name__ == "__main__":
    # Sample job postings
    sample_postings = [
            """            
A nsys empowers the world's most innovative companies to design and deliver transformational products by offering the best and broadest engineering simulation software to solve the most complex design challenges and engineer products limited only by imagination. Thus, through our enriching internship program, we help develop the next-generation of engineers and technologists.

As a student intern/co-op, you will help develop our industry-leading simulation software while gaining real experience in your field of study. You will work on a variety of impactful projects to help maintain, advance, and accelerate Ansys products. In this internship, you will support our Electronics Business Unit. This is a 40-hour per week paid position starting August 2025 and concluding December 2025. This internship is fully remote.

Responsibilities

     Work alongside software developers during the design, implementation, and verification process of the SIwave product.
     Work on implementing front-end and back-end modules using modern web technologies.
     Work on developing/improving the existing post-analysis visualization software modules.
     Investigate defects in production code.
     Develop unit and regression tests.
     Understand and employ best software practice.

Minimum Qualifications

     Pursuing a Master’s in Computer Science, Engineering, or related technical degree.
     Basic understanding of Object-Oriented Programming.
     Basic knowledge of one of these languages: C++, JavaScript, Typescript
     Good knowledge of data-structures and algorithm.
     Basic knowledge of full-stack development: Front-end and back-end

Preferred Qualifications

     Experience with software development is a plus.
     Knowledge of these tools is a plus: React JS, Node, HTML/CSS, Angular

At Ansys, we know that changing the world takes vision, skill, and each other. We fuel new ideas, build relationships, and help each other realize our greatest potential. We are ONE Ansys. We operate on three key components: the commitments to our stakeholders, the behaviors of how we work together, and the actions of how we deliver results. Together as ONE Ansys, we are powering innovation that drives human advancement. 


"""
    ]
    
    # Initialize and run classifier
    classifier = techJobClassifier()
    results_df = classifier.process_batch(sample_postings)
    print(results_df)
