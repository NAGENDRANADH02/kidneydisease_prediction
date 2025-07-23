from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import render_to_string
from .forms import CKDForm
import joblib
import os
from xhtml2pdf import pisa
from io import BytesIO
from datetime import datetime
model_path = os.path.join(os.path.dirname(__file__), 'CKD.pkl')
model = joblib.load(model_path)

def index(request):
    if request.method == 'POST':
        form = CKDForm(request.POST)
        if form.is_valid():
            features = list(form.cleaned_data.values())

            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0][1]
            
            result = "CKD Detected" if prediction == 1 else "No CKD"
            risk_score = f"{probability * 100:.2f}%"

            return render(request, 'predictor/result.html', {
                'result': result,
                'risk_score': risk_score,
                'form_data': form.cleaned_data
            })
    else:
        form = CKDForm()
    
    return render(request, 'predictor/index.html', {'form': form})


def generate_pdf(request):
    if request.method == 'POST':
        form_data = request.POST.dict()
        form_data.pop('csrfmiddlewaretoken', None) 

        result = form_data.get('result', 'N/A')
        risk_score = form_data.get('risk_score', 'N/A')

    
        formatted_data = {
            key.replace('_', ' ').capitalize(): value
            for key, value in form_data.items()
            if key not in ['result', 'risk_score']
        }
        context = {
            'form_data': formatted_data,
            'result': result,
            'risk_score': risk_score,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        html = render_to_string('predictor/pdf.html', context)

    
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="CKD_Report.pdf"'

        pisa_status = pisa.CreatePDF(BytesIO(html.encode('UTF-8')), dest=response)
        if pisa_status.err:
            return HttpResponse('Error generating PDF', status=500)
        return response
