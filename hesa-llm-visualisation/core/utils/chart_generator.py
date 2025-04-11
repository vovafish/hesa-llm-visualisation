# import matplotlib.pyplot as plt
# import io
# import base64

# def generate_chart(data, chart_type='bar'):
#     """
#     Generate chart from processed data
#     """
#     plt.figure(figsize=(10, 6))
    
#     if chart_type == 'bar':
#         # Example bar chart
#         plt.bar(data.index, data.values)
    
#     # Save to bytes buffer
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     image_png = buffer.getvalue()
#     buffer.close()
    
#     # Encode
#     graphic = base64.b64encode(image_png)
#     return graphic.decode('utf-8')
