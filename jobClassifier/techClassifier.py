import pandas
import re
from collections import defaultdict
import json
from transformers import BartTokenizerFast, BartForConditionalGeneration

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
            summary = self.summarize(posting)
            print(summary)
            
            # Flatten the results for CSV format
            flat_result = {
                'posting_id': idx,
                'programming_languages': ', '.join(classification.get('programming_languages', [])),
                'frameworks_libraries': ', '.join(classification.get('frameworks_libraries', [])),
                'databases': ', '.join(classification.get('databases', [])),
                'cloud_platforms': ', '.join(classification.get('cloud_platforms', [])),
                'tools': ', '.join(classification.get('tools', [])),
                'AI_summary': summary
            }
            results.append(flat_result)
        
        # Convert to DataFrame and save
        df = pandas.DataFrame(results)
        df.to_csv(output_file, index=False)
        return df

    def summarize(self, text):
        print("Starting AI summarization")

        tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large-cnn")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

        inputs = tokenizer(text, max_length = 1024, truncation = True, return_tensors = 'pt')
        # Length_penalty >1 encourages longer outputs
        summary_ids = model.generate(inputs['input_ids'], min_length = 100, max_length = 300, length_penalty = 2.0, num_beams = 4, early_stopping = True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
        
        return summary

if __name__ == "__main__":
    # Sample job postings
    sample_postings = [
            r"""
            About Kinaxis

In 1984, we started out as a team of three engineers. Today, we have grown to become a global organization with over 2000 employees around the world, with a brand-new HQ based in Kanata North in Ottawa. As one of Canada’s Top Employers for Young People (2024), we want to help you take the first step in crafting your career journey.

At Kinaxis, we power the world’s supply chains to help preserve the planet’s resources and enrich the human experience. As a global leader in end-to-end supply chain management, we enable supply chain excellence for all industries, with more than 40,000 users in over 100 countries. We are expanding our team as we continue to innovate and revolutionize how we support our customers.

Location

This is a hybrid position. You must be in the Ottawa, Canada office, at least three days a week.

About The Team

Kinaxis is looking for a talented candidate to work within the Core Algorithms Development team. The team is responsible for developing various algorithms to solve supply chain management problems. The uniqueness of the team is that it performs at the intersection of technology and real business problems. You will contribute to the product that delights customers world-wide! If you are someone that has a passion for technology and are looking for an opportunity to learn and grow—this challenge is for you.

 This is a full-time, 8 or 12-month position, starting May 2025. 

 To be eligible for a Co-op or Intern position at Kinaxis, you must either be currently enrolled in full-time education or, if you are a recent/upcoming graduate, your graduation date must be within 12 months of the placement end date. 

 What you will do 

     Work under the supervision of a mentor who will guide you through your journey at Kinaxis 
     Work together with other developers and tester on feature development and performance enhancements 
     Maintain and develop tools for testing and debugging 
     Collaborate and work closely with your team members 
     May perform additional projects upon request 

 What we are looking for 

     Completion of 1st year of studies in a Computer Science or Software Engineering (or equivalent) program 
     C++ programming experience required. 
     Familiarity with Algorithms and Data Structure 
     Desire to enhance your skills in Object-Oriented Design and related best practices 
     Very strong technical aptitude, with a preference for algorithms, data structures 
     Insatiable drive for improving performance – you want to do it fast and right! 
     Good collaborative skills and positive attitude 
     Excellent verbal and written communications skills 
     Strong initiative and drive to get things done! 
     Exposure and willingness to develop under agile framework and software development lifecycle processes 

We’re accepting applications now through end of day on Thursday, Feb 6, 2025. Please note that we may begin reviewing applications before the posting closes, so early submission is encouraged.

#Coop, #Internship, #Intern, 

Work With Impact: Our platform directly helps companies power the world’s supply chains. We see the results of what we do out in the world every day—when we see store shelves stocked, when medications are available for our loved ones, and so much more.

Work with Fortune 500 Brands: Companies across industries trust us to help them take control of their integrated business planning and digital supply chain. Some of our customers include Ford, Unilever, Yamaha, P&G, Lockheed-Martin, and more.

 Social Responsibility at Kinaxis:  Our Diversity, Equity, and Inclusion Committee weighs in on hiring practices, talent assessment training materials, and mandatory training on unconscious bias and inclusion fundamentals. Sustainability is key to what we do and we’re committed to net-zero operations strategy for the long term. We are involved in our communities and support causes where we can make the most impact.

People matter at Kinaxis and these are some of the perks and benefits we created for our team:

     Flexible vacation and Kinaxis Days (company-wide day off on the last Friday of every month) 
     Flexible work options 
     Physical and mental well-being programs 
     Regularly scheduled virtual fitness classes 
     Mentorship programs and training and career development 
     Recognition programs and referral rewards 
     Hackathons 

For more information, visit the Kinaxis web site at www.kinaxis.com or the company’s blog at http://blog.kinaxis.com .

Kinaxis welcomes candidates to apply to our inclusive community. We provide accommodations upon request to ensure fairness and accessibility throughout our recruitment process for all candidates, including those with specific needs or disabilities. If you require an accommodation, please reach out to us at recruitmentprograms@kinaxis.com. Please note that this contact information is strictly for accessibility requests and cannot be used to inquire about application statuses.

Kinaxis is committed to ensuring a fair and transparent recruitment process. We use artificial intelligence (AI) tools in the initial step of the recruitment process to compare submitted resumes against the job description, to identify candidates whose education, experience and skills most closely match the requirements of the role. After the initial screening, all subsequent decisions regarding your application, including final selection, are made by our human recruitment team. AI does not make any final hiring decisions.
""",
            r"""            
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
