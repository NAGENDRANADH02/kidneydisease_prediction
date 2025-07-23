# predictor/forms.py

from django import forms

class CKDForm(forms.Form):
    specific_gravity = forms.FloatField()
    red_blood_cell_count = forms.FloatField()
    hemoglobin = forms.FloatField()
    hypertension = forms.IntegerField()
    diabetesmellitus = forms.IntegerField()
    albumin = forms.FloatField()
    packed_cell_volume = forms.FloatField()
    appetite = forms.IntegerField()
    sodium = forms.FloatField()
    pus_cell = forms.IntegerField()
    pedal_edema = forms.IntegerField()
    blood_urea = forms.FloatField()
    blood_glucose_random = forms.FloatField()
    anemia = forms.IntegerField()
    sugar = forms.FloatField()
    blood_pressure = forms.FloatField()
    serum_creatinine = forms.FloatField()
    red_blood_cells = forms.IntegerField()
    pus_cell_clumps = forms.IntegerField()
    coronary_artery_disease = forms.IntegerField()
    age = forms.FloatField()
    white_blood_cell_count = forms.FloatField()
