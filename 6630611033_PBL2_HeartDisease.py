import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go

attribute_descriptions = {
    'age': 'อายุ (ปี)',
    'sex': 'เพศ (0 = หญิง, 1 = ชาย)',
    'cp': 'ประเภทอาการเจ็บหน้าอก (0 = ไม่เจ็บ, 1 = เจ็บแบบไม่คงที่, 2 = เจ็บแบบไม่เกี่ยวกับหัวใจ, 3 = เจ็บแบบคงที่)',
    'trestbps': 'ความดันโลหิตขณะพัก (มม.ปรอท)',
    'chol': 'คอเลสเตอรอลในเลือด (มก./ดล.)',
    'fbs': 'น้ำตาลในเลือดขณะงดอาหาร > 120 มก./ดล. (0 = ไม่ใช่, 1 = ใช่)',
    'restecg': 'ผลคลื่นไฟฟ้าหัวใจขณะพัก (0 = ปกติ, 1 = ผิดปกติแบบ ST-T, 2 = มีภาวะกล้ามเนื้อหัวใจโต)',
    'thalach': 'อัตราการเต้นของหัวใจสูงสุดที่วัดได้ (ครั้ง/นาที)',
    'exang': 'มีอาการเจ็บหน้าอกเมื่อออกกำลังกาย (0 = ไม่มี, 1 = มี)',
    'oldpeak': 'การเปลี่ยนแปลงของคลื่น ST เมื่อออกกำลังกายเทียบกับขณะพัก',
    'slope': 'ความชันของส่วน ST ขณะออกกำลังกาย (1 = ขึ้น, 2 = ราบ, 3 = ลง)',
    'ca': 'จำนวนหลอดเลือดหลักที่มีการตีบ (0-3)',
    'thal': 'ความผิดปกติของกล้ามเนื้อหัวใจ (1 = ปกติ, 2 = มีความผิดปกติแบบคงที่, 3 = มีความผิดปกติแบบไม่คงที่)',
    'target': 'การวินิจฉัยโรคหัวใจ (0 = ไม่เป็นโรคหัวใจ, 1 = เป็นโรคหัวใจ)'
}

# Load dataset
try:
    df = pd.read_csv("heart.csv")
except FileNotFoundError:
    print("ข้อผิดพลาด: ไม่พบไฟล์ heart.csv กรุณาตรวจสอบว่าไฟล์อยู่ในไดเรกทอรีปัจจุบัน")
    import sys
    sys.exit(1)

# Create more descriptive target labels
df['target_label'] = df['target'].map({0: "0 (ไม่เป็นโรคหัวใจ)", 1: "1 (เป็นโรคหัวใจ)"})

# Data overview
print(f"ขนาดของชุดข้อมูล: {df.shape}")
print("\nข้อมูล 5 แถวแรก:")
print(df.head())
print("\nประเภทของข้อมูล:")
print(df.dtypes)
print("\nข้อมูลที่ขาดหาย:")
print(df.isnull().sum())

# Data preprocessing
# Check for outliers and handle them if necessary
Q1 = df.select_dtypes(include=['number']).quantile(0.25)
Q3 = df.select_dtypes(include=['number']).quantile(0.75)
IQR = Q3 - Q1
outliers = ((df.select_dtypes(include=['number']) < (Q1 - 1.5 * IQR)) | (df.select_dtypes(include=['number']) > (Q3 + 1.5 * IQR))).sum()
print("\nจำนวนค่าผิดปกติในแต่ละคอลัมน์:")
print(outliers)

# Prepare data for classification
X = df.drop(['target', 'target_label'], axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Classification models with cross-validation
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),  # Increased max_iter to ensure convergence
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = []
conf_matrices = {}
class_reports = {}

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    
    # Train on training set
    model.fit(X_train, y_train)
    
    # Test on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    test_acc = np.round((y_pred == y_test).mean(), 4)
    cv_acc = np.round(cv_scores.mean(), 4)
    
    # Store results
    results.append({
        "โมเดล": name, 
        "Test Accuracy": test_acc,
        "Cross-Validation Accuracy": cv_acc
    })
    
    # Confusion matrix
    conf_matrices[name] = confusion_matrix(y_test, y_pred)
    
    # Classification report
    class_reports[name] = classification_report(y_test, y_pred, output_dict=True)

# Feature importance for Random Forest
rf_model = models["Random Forest"]
feature_importance = pd.DataFrame({
    'คุณลักษณะ': X.columns,
    'ความสำคัญ': rf_model.feature_importances_
}).sort_values('ความสำคัญ', ascending=False)

# Add descriptions to feature importance
feature_importance['คำอธิบาย'] = feature_importance['คุณลักษณะ'].map(lambda x: attribute_descriptions.get(x, ''))

# Clustering with PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Find optimal number of clusters using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Apply KMeans with optimal number of clusters (let's assume 3 based on visualization)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df_clustered = df.copy()
df_clustered['กลุ่ม'] = clusters

# Update cluster visualization
cluster_fig = px.scatter(
    x=X_pca[:,0], y=X_pca[:,1], 
    color=df_clustered['กลุ่ม'].astype(str),
    labels={'x': 'PCA 1', 'y': 'PCA 2', 'color': 'กลุ่ม'},
    title="การแสดงผลการทำ Cluster ด้วย KMeans (ลดมิติด้วย PCA)",
    template="plotly_white",  # Use a nicer template
    color_discrete_sequence=['#3498db', '#9b59b6', '#f39c12']  # สีฟ้า, ม่วง, ส้ม - แยกกลุ่มชัดเจน
)

# Cluster analysis
cluster_analysis = df_clustered.groupby('กลุ่ม').mean(numeric_only=True).reset_index()
cluster_analysis_rounded = cluster_analysis.round(2)

# Association Rules - Fix for the value error
df_ar = df.copy()

# First, convert all features to binary format properly
for col in df_ar.columns:
    if col not in ['target', 'target_label']:
        if df_ar[col].nunique() > 2:
            col_median = df_ar[col].median()
            col_name = f"{col}_high"
            df_ar[col_name] = (df_ar[col] > col_median).astype(int)
            # Remove the original column
            df_ar = df_ar.drop(col, axis=1)
        else:
            df_ar[col] = df_ar[col].astype(int)

# Drop the target_label column for association rules
df_ar = df_ar.drop('target_label', axis=1, errors='ignore')

# Display a sample of the prepared data to verify it's binary
print("Sample of prepared data for association rules:")
print(df_ar.head())
print(f"Unique values per column: {[df_ar[col].unique() for col in df_ar.columns]}")

# Generate frequent itemsets and rules
try:
    # Check that all values are binary
    for col in df_ar.columns:
        unique_vals = df_ar[col].unique()
        if not all(val in [0, 1] for val in unique_vals):
            print(f"Warning: Column {col} contains non-binary values: {unique_vals}")
            # Convert to strictly binary values
            df_ar[col] = df_ar[col].apply(lambda x: 1 if x > 0 else 0)
    
    frequent_itemsets = apriori(df_ar, min_support=0.05, use_colnames=True)
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
        rules = rules.sort_values('confidence', ascending=False)
        
        # Format rules for display
        # Find this part in your code
        rules_df = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20).copy()
        rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_df = rules_df.round(3)
        
        # Translate column names
        rules_df.columns = ['เงื่อนไข', 'ผลลัพธ์', 'support', 'confidence', 'lift']
    else:
        print("ไม่พบรูปแบบที่เกิดบ่อยด้วยค่าขีดจำกัดปัจจุบัน")
        rules_df = pd.DataFrame(columns=['เงื่อนไข', 'ผลลัพธ์', 'support', 'confidence', 'lift'])
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการทำเหมืองกฎความสัมพันธ์: {e}")
    # Print more details about the exception
    import traceback
    traceback.print_exc()
    rules_df = pd.DataFrame(columns=['เงื่อนไข', 'ผลลัพธ์', 'support', 'confidence', 'lift'])

# Create a function to get display names for features
def get_display_name(col):
    if col in attribute_descriptions:
        return f"{col} ({attribute_descriptions[col]})"
    return col

# Dash app with improved styling
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "แดชบอร์ดวิเคราะห์โรคหัวใจด้วย ML"

# Define colors and styling
colors = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'accent': '#3498db',          # สีฟ้า - สำหรับข้อมูลทั่วไป
    'accent2': '#9b59b6',         # สีม่วง - สำหรับหัวข้อรอง
    'accent3': '#2ecc71',         # สีเขียว - สำหรับผลลัพธ์เชิงบวก
    'negative': '#e74c3c',        # สีแดง - สำหรับผลลัพธ์เชิงลบ
    'warning': '#f39c12',         # สีส้ม - สำหรับคำเตือน
    'grid': '#ecf0f1'
}

# Custom CSS for better styling
external_scripts = ['/assets/custom.js']
external_stylesheets = ['/assets/custom.css']

app = dash.Dash(__name__,
                external_scripts=external_scripts,
                external_stylesheets=external_stylesheets)

# Define a reusable card component function
def create_card(title, content, extra_class="", style=None):
    card_style = {
        "padding": "20px", 
        "backgroundColor": "#fff", 
        "borderRadius": "8px", 
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "marginBottom": "20px"
    }
    
    # Merge additional style if provided
    if style:
        card_style.update(style)
    
    return html.Div([
        html.H3(title, style={"color": colors['accent'], "marginBottom": "15px", "paddingBottom": "10px", "borderBottom": "1px solid #eee"}),
        html.Div(content)
    ], className=f"dashboard-card {extra_class}", style=card_style)

app.layout = html.Div([
    html.Div([
        html.H1("Heart Disease ML Dashboard", 
            className="fade-in",
            style={"textAlign": "center", "marginBottom": "30px", "marginTop": "20px"})
    ]),
    
    dcc.Tabs([
        dcc.Tab(label='Classification', style={'backgroundColor': colors['background']}, children=[
            html.Div([
                create_card(
                    "ประสิทธิภาพของโมเดล",
                    [
                        html.P("เปรียบเทียบอัลกอริทึมการจำแนกประเภทต่างๆ สำหรับการทำนายโรคหัวใจ", 
                               className="text-muted"),
                        
                        # Model accuracy comparison plot
                        dcc.Graph(
                            figure=px.bar(
                                pd.DataFrame(results), 
                                x="โมเดล", 
                                y=["Test Accuracy", "Cross-Validation Accuracy"],
                                barmode="group",
                                title="เปรียบเทียบความแม่นยำของโมเดล",
                                template="plotly_white",
                                color_discrete_sequence=['#3498db', '#2ecc71']
                            ).update_layout(
                                xaxis_title="โมเดล",
                                yaxis_title="คะแนนความแม่นยำ",
                                legend_title="ประเภทข้อมูล",
                                plot_bgcolor='#f8f9fa'
                            )
                        )
                    ]
                ),
                
                create_card(
                    "ความสำคัญของคุณลักษณะ (Random Forest)",
                    [
                        # Feature importance table with descriptions
                        dash_table.DataTable(
                            feature_importance[['คุณลักษณะ', 'คำอธิบาย', 'ความสำคัญ']].to_dict('records'),
                            columns=[
                                {"name": "คุณลักษณะ", "id": "คุณลักษณะ"},
                                {"name": "คำอธิบาย", "id": "คำอธิบาย"},
                                {"name": "ความสำคัญ", "id": "ความสำคัญ", "format": {"specifier": ".4f"}}
                            ],
                            style_table={'overflowX': 'auto'},
                            style_header={
                                'backgroundColor': '#34495e',
                                'color': 'white',
                                'fontWeight': 'bold'
                            },
                            style_cell={
                                'textAlign': 'left',
                                'padding': '10px'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#f8f9fa'
                                }
                            ]
                        ),
                        
                        # Bar chart for feature importance
                        dcc.Graph(
                            figure=px.bar(
                                feature_importance,
                                x='ความสำคัญ',
                                y='คุณลักษณะ',
                                orientation='h',
                                title="ความสำคัญของคุณลักษณะจาก Random Forest",
                                template="plotly_white",
                                color='ความสำคัญ',
                                color_continuous_scale='RdYlBu_r'
                            ).update_layout(
                                yaxis={'categoryorder': 'total ascending'},
                                plot_bgcolor='#f8f9fa'
                            )
                        )
                    ]
                ),
                
                create_card(
                    "Confusion Matrix",
                    [
                        html.P("เลือกโมเดลเพื่อดู Confusion Matrix ตาม Model ต่างๆ:", style={"fontStyle": "italic"}),
                        
                        dcc.Dropdown(
                            id='confusion-matrix-dropdown',
                            options=[{'label': name, 'value': name} for name in models.keys()],
                            value=list(models.keys())[0],
                            style={"marginBottom": "20px", "width": "50%"}
                        ),
                        
                        dcc.Graph(id='confusion-matrix-graph')
                    ]
                )
            ], style={"padding": "20px", "backgroundColor": colors['background']})
        ]),
        
        dcc.Tab(label='Clustering', style={'backgroundColor': colors['background']}, children=[
            html.Div([
                html.H2("การวิเคราะห์การ Clustering ด้วย K-Means", style={"color": colors['accent'], "marginTop": "20px"}),
                html.P("การจัดกลุ่มผู้ป่วยตามคุณลักษณะ", style={"fontStyle": "italic"}),
                
                create_card(
                    "Cluster ที่เหมาะสม",
                    [
                        # Elbow method plot
                        dcc.Graph(
                            figure=px.line(
                                x=list(range(1, 11)), 
                                y=wcss,
                                markers=True,
                                title="Elbow Method สำหรับหาค่า K ที่เหมาะสมในโมเดล KMeans",
                                template="plotly_white",
                                labels={'x': 'จำนวนกลุ่ม', 'y': 'WCSS'}
                            ).update_layout(xaxis_title="จำนวนกลุ่ม", yaxis_title="ผลรวมกำลังสองภายใน cluster")
                        )
                    ]
                ),
                
                create_card(
                    "Cluster",
                    [
                        # Cluster visualization
                        dcc.Graph(figure=cluster_fig)
                    ]
                ),
                
                create_card(
                    "Cluster Profiles",
                    [
                        html.P("ค่าเฉลี่ยของ Cluster Profiles ในแต่ละกลุ่ม", style={"fontStyle": "italic"}),
                        
                        # Explanation of attributes
                        html.Div([
                            html.H4("คำอธิบายคุณลักษณะ:", style={"color": colors['accent2'], "marginTop": "15px"}),
                            html.Ul([
                                html.Li([
                                    html.Strong(f"{attr}: "), 
                                    html.Span(desc)
                                ]) for attr, desc in attribute_descriptions.items()
                            ])
                        ], className="feature-description", style={"backgroundColor": "#f5f5f5", "padding": "15px", "borderRadius": "5px", "marginBottom": "15px"}),
                        
                        dash_table.DataTable(
                            cluster_analysis_rounded.to_dict('records'),
                            columns=[{"name": col if col == 'กลุ่ม' else f"{col}", "id": col} for col in cluster_analysis_rounded.columns],
                            style_table={'overflowX': 'auto'},
                            style_header={
                                'backgroundColor': colors['accent'],
                                'color': 'white',
                                'fontWeight': 'bold'
                            },
                            style_cell={
                                'textAlign': 'center',
                                'padding': '10px'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#f8f9fa'
                                }
                            ],
                            tooltip_data=[
                                {
                                    column: {'value': attribute_descriptions.get(column, '')}
                                    for column in cluster_analysis_rounded.columns if column != 'กลุ่ม'
                                } for _ in range(len(cluster_analysis_rounded))
                            ],
                            tooltip_duration=None
                        )
                    ]
                ),
                
                create_card(
                    "การกระจายของเป้าหมายในแต่ละ clusters",
                    [
                        # Target distribution in clusters
                        dcc.Graph(
                            figure=px.histogram(
                                df_clustered, 
                                x="กลุ่ม", 
                                color="target_label",
                                barmode="group",
                                title="การกระจายของโรคหัวใจในแต่ละกลุ่ม",
                                template="plotly_white",
                                color_discrete_sequence=['#2ecc71', '#e74c3c']
                            ).update_layout(
                                xaxis_title="กลุ่ม",
                                yaxis_title="จำนวน",
                                legend_title="โรคหัวใจ",
                                plot_bgcolor='#f8f9fa'
                            )
                        )
                    ]
                )
            ], style={"padding": "20px", "backgroundColor": colors['background']})
        ]),
        
        dcc.Tab(label='Association Rules', style={'backgroundColor': colors['background']}, children=[
            html.Div([
                html.H2("การวิเคราะห์ Association Rules", style={"color": colors['accent'], "marginTop": "20px"}),
                html.P("การค้นพบรูปแบบและความสัมพันธ์ในข้อมูลโรคหัวใจ", style={"fontStyle": "italic"}),
                
                create_card(
                    "คำอธิบายคุณลักษณะ",
                    [
                        # Explanation of attributes for association rules
                        html.Ul([
                            html.Li([
                                html.Strong(f"{attr}: "), 
                                html.Span(desc)
                            ]) for attr, desc in attribute_descriptions.items()
                        ])
                    ],
                    style={"backgroundColor": "#f5f5f5"}
                ),
                
                create_card(
                    "กฎความสัมพันธ์ที่สำคัญ",
                    [
                        html.P("กฎเรียงตามค่า confidence (Minimum support: 5%, Minimum confidence: 60%)", 
                               style={"fontStyle": "italic"}),
                        
                        dash_table.DataTable(
                            rules_df.to_dict('records'),
                            columns=[{"name": col, "id": col} for col in rules_df.columns],
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_header={
                                'backgroundColor': colors['accent'],
                                'color': 'white',
                                'fontWeight': 'bold'
                            },
                            style_cell={
                                'textAlign': 'left',
                                'padding': '10px'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#f8f9fa'
                                }
                            ],
                            tooltip_header={
                                'support': 'ความถี่ของการเกิดกฎในชุดข้อมูล',
                                'confidence': 'ความน่าจะเป็นแบบมีเงื่อนไขของผลลัพธ์เมื่อกำหนดเงื่อนไข',
                                'lift': 'อัตราส่วนของค่าสนับสนุนที่สังเกตได้ต่อค่าสนับสนุนที่คาดหวังหากตัวแปรเป็นอิสระต่อกัน'
                            }
                        )
                    ]
                ),
                
                create_card(
                    "Association Rule",
                    [
                        html.P("ค่า support และค่า confidence โดยใช้ค่ายกเป็นสี", style={"fontStyle": "italic"}),
                        
                        dcc.Graph(
                            figure=px.scatter(
                                rules_df if not rules_df.empty else pd.DataFrame(),
                                x="support",
                                y="confidence",
                                size="lift",
                                color="lift",
                                hover_name="ผลลัพธ์",
                                size_max=20,
                                title="ค่า support และค่า confidence ของ Association Rules",
                                template="plotly_white",
                                color_continuous_scale='YlOrRd'
                            ).update_layout(
                                xaxis_title="support",
                                yaxis_title="confidence",
                                plot_bgcolor='#f8f9fa'
                        )
                    ) if not rules_df.empty else html.P("ไม่มีกฎที่จะแสดงด้วยค่าขีดจำกัดปัจจุบัน")
                    ]
                )
            ], style={"padding": "20px", "backgroundColor": colors['background']})
        ]),
        
        dcc.Tab(label='Data Overview', style={'backgroundColor': colors['background']}, children=[
            html.Div([
                html.H2("การสำรวจชุดข้อมูล", style={"color": colors['accent'], "marginTop": "20px"}),
                html.P("ภาพรวมและสถิติของชุดข้อมูลโรคหัวใจ", style={"fontStyle": "italic"}),
                
                create_card(
                    "คำอธิบายคุณลักษณะ",
                    [
                        html.Div([
                            html.Table([
                                html.Thead(
                                    html.Tr([
                                        html.Th('คุณลักษณะ', style={'width': '15%', 'textAlign': 'left', 'padding': '8px', 'borderBottom': '1px solid #ddd'}),
                                        html.Th('คำอธิบาย', style={'width': '85%', 'textAlign': 'left', 'padding': '8px', 'borderBottom': '1px solid #ddd'})
                                    ])
                                ),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(attr, style={'padding': '8px', 'borderBottom': '1px solid #ddd', 'fontWeight': 'bold'}),
                                        html.Td(desc, style={'padding': '8px', 'borderBottom': '1px solid #ddd'})
                                    ], style={'backgroundColor': '#f9f9f9' if i % 2 else 'white'}) 
                                    for i, (attr, desc) in enumerate(attribute_descriptions.items())
                                ])
                            ], style={'width': '100%', 'borderCollapse': 'collapse'})
                        ])
                    ],
                    style={"backgroundColor": "#f5f5f5"}
                ),
                
                create_card(
                    "สรุปชุดข้อมูล",
                    [
                        html.P(f"จำนวนตัวอย่าง: {df.shape[0]}", style={"fontSize": "16px"}),
                        html.P(f"จำนวนคุณลักษณะ: {df.shape[1]-2}", style={"fontSize": "16px"}),  # -2 to account for target and target_label
                        html.P(f"การกระจายของเป้าหมาย: เป็นโรคหัวใจ {df['target'].value_counts()[1]} คน, ไม่เป็นโรคหัวใจ {df['target'].value_counts()[0]} คน", 
                               style={"fontSize": "16px"}),
                    ]
                ),
                
                create_card(
                    "ตัวอย่างข้อมูล",
                    [
                        dash_table.DataTable(
                            df.head(10).to_dict('records'),
                            columns=[{"name": col, "id": col} for col in df.columns],
                            style_table={'overflowX': 'auto'},
                            style_header={
                                'backgroundColor': colors['accent'],
                                'color': 'white',
                                'fontWeight': 'bold'
                            },
                            style_cell={
                                'textAlign': 'center',
                                'padding': '10px'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': '#f8f9fa'
                                }
                            ]
                        )
                    ]
                ),
                
                create_card(
                    "การกระจายของคุณลักษณะ",
                    [
                        html.P("เลือกคุณลักษณะเพื่อดูการกระจาย:", style={"fontStyle": "italic"}),
                        
                        dcc.Dropdown(
                            id='feature-dropdown',
                            options=[{'label': col, 'value': col} for col in df.columns if col not in ['target', 'target_label']],
                            value=df.columns[0] if 'target' != df.columns[0] else df.columns[1],
                            style={"marginBottom": "20px", "width": "50%"}
                        ),
                        
                        dcc.Graph(id='feature-distribution')
                    ]
                )
            ], style={"padding": "20px", "backgroundColor": colors['background']})
        ])
    ], style={
        'fontFamily': 'Arial, sans-serif',
        'borderRadius': '5px',
        'backgroundColor': colors['background']
    })
], style={
    'maxWidth': '1200px',
    'margin': '0 auto',
    'backgroundColor': colors['background'],
    'padding': '20px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'fontFamily': 'Arial, sans-serif'
})

# Callbacks for interactive elements
@app.callback(
    Output('confusion-matrix-graph', 'figure'),
    [Input('confusion-matrix-dropdown', 'value')]
)
def update_confusion_matrix(selected_model):
    cm = conf_matrices[selected_model]
    
    # Create heatmap with updated labels
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="True", color="จำนวน"),
        x=['0 (ไม่เป็นโรคหัวใจ)', '1 (เป็นโรคหัวใจ)'],
        y=['0 (ไม่เป็นโรคหัวใจ)', '1 (เป็นโรคหัวใจ)'],
        text_auto=True,
        color_continuous_scale='RdBu_r',  # สีแดง-น้ำเงิน เหมาะสำหรับ confusion matrix
        title=f"Confusion Matrix - {selected_model}",
        template="plotly_white"
    )
    
    return fig

@app.callback(
    Output('feature-distribution', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_feature_distribution(selected_feature):
    # Create histogram with target color - using target_label for better labels
    fig = px.histogram(
        df,
        x=selected_feature,
        color="target_label", 
        barmode="overlay",
        opacity=0.7,
        title=f"การกระจายของ {selected_feature} ตามสถานะโรคหัวใจ",
        template="plotly_white",
        color_discrete_sequence=['#2ecc71', '#e74c3c']  # สีเขียว (ไม่เป็นโรค) และสีแดง (เป็นโรค)
    )
    
    fig.update_layout(
        xaxis_title=selected_feature,
        yaxis_title="จำนวน",
        legend_title="โรคหัวใจ"
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)