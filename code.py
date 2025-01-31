import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from google.colab import files
import io

class ShoppingTrendsAnalyzer:
    def __init__(self, dataframe):
        """
        Initialize the analyzer with shopping trends data

        Parameters:
        dataframe (pd.DataFrame): DataFrame containing shopping data
        """
        self.df = dataframe

    def preprocess_data(self):
        """
        Preprocess and clean the data
        """
        # Convert data types if needed
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')
        self.df['Quantity'] = pd.to_numeric(self.df['Quantity'], errors='coerce')

        # Create new features
        self.df['Total_Purchase_Value'] = self.df['Price'] * self.df['Quantity']

        # Age categorization
        self.df['Age_Group'] = pd.cut(self.df['Age'],
            bins=[0, 20, 30, 40, 50, 100],
            labels=['Under 20', '20-30', '30-40', '40-50', 'Over 50']
        )

    def comprehensive_analysis(self):
        """
        Perform comprehensive shopping trend analysis
        """
        # Suppress warnings
        warnings.filterwarnings('ignore')

        # Preprocess data first
        self.preprocess_data()

        # 1. Detailed Sales Analysis
        print("=== COMPREHENSIVE SALES ANALYSIS ===")

        # Category Performance
        category_performance = self.df.groupby('Category').agg({
            'Total_Purchase_Value': ['sum', 'mean', 'count'],
            'Quantity': 'sum'
        }).reset_index()
        category_performance.columns = ['Category', 'Total_Sales', 'Avg_Sale_Value', 'Transaction_Count', 'Total_Quantity_Sold']
        category_performance = category_performance.sort_values('Total_Sales', ascending=False)
        print("\nCategory Performance:")
        print(category_performance)

        # 2. Demographic Insights
        demographic_analysis = self.df.groupby(['Age_Group', 'Gender']).agg({
            'Total_Purchase_Value': ['sum', 'mean'],
            'Quantity': 'sum'
        }).reset_index()
        demographic_analysis.columns = ['Age_Group', 'Gender', 'Total_Spend', 'Avg_Purchase', 'Total_Quantity']
        print("\nDemographic Spending Insights:")
        print(demographic_analysis)

        # Create visualizations
        self._create_visualizations()

    def _create_visualizations(self):
        """
        Create advanced data visualizations
        """
        plt.figure(figsize=(20, 15))

        # 1. Total Sales by Category with Error Bars
        plt.subplot(2, 2, 1)
        category_sales = self.df.groupby('Category')['Total_Purchase_Value'].agg(['sum', 'std'])
        plt.bar(category_sales.index, category_sales['sum'], yerr=category_sales['std'], capsize=10)
        plt.title('Total Sales by Category with Variance')
        plt.xlabel('Category')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)

        # 2. Purchase Distribution by Age Group and Gender
        plt.subplot(2, 2, 2)
        age_gender_spend = self.df.groupby(['Age_Group', 'Gender'])['Total_Purchase_Value'].sum().unstack()
        age_gender_spend.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Purchase Distribution by Age Group and Gender')
        plt.xlabel('Age Group')
        plt.ylabel('Total Purchase Value')
        plt.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')

        # 3. Box Plot of Prices by Category
        plt.subplot(2, 2, 3)
        sns.boxplot(x='Category', y='Price', data=self.df)
        plt.title('Price Distribution by Category')
        plt.xticks(rotation=45)

        # 4. Correlation Heatmap
        plt.subplot(2, 2, 4)
        correlation_cols = ['Age', 'Price', 'Quantity', 'Total_Purchase_Value']
        correlation_matrix = self.df[correlation_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')

        plt.tight_layout()
        plt.show()

    def generate_insights(self):
        """
        Generate insights and recommendations
        """
        print("\n=== INSIGHTS & RECOMMENDATIONS ===")

        # Top performing categories
        top_categories = self.df.groupby('Category')['Total_Purchase_Value'].sum().nlargest(3)
        print("Top 3 Performing Categories:")
        print(top_categories)

        # Purchase behavior by gender
        gender_behavior = self.df.groupby('Gender').agg({
            'Total_Purchase_Value': ['mean', 'sum'],
            'Quantity': ['mean', 'sum']
        })
        print("\nGender-based Purchase Behavior:")
        print(gender_behavior)

def upload_and_analyze():
    """
    Upload CSV file and perform analysis in Google Colab
    """
    print("Please upload your CSV file.")

    # Upload file
    uploaded = files.upload()

    # Check if file was uploaded
    if not uploaded:
        print("No file uploaded. Exiting.")
        return

    # Read the uploaded file
    filename = list(uploaded.keys())[0]
    df = pd.read_csv(io.BytesIO(uploaded[filename]))

    # Create analyzer and run analysis
    analyzer = ShoppingTrendsAnalyzer(df)

    # Perform comprehensive analysis
    analyzer.comprehensive_analysis()

    # Generate insights
    analyzer.generate_insights()

# Run the analysis
if __name__ == '__main__':
    upload_and_analyze()
