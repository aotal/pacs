import pydicom

filepath = "output_processed_dicom/Img1_SN2133ED_KVP73_mAs1.98_IE95.dcm"
ds = pydicom.dcmread(filepath)

print(f"PatientID: {ds.get('PatientID', 'N/A')}")
print(f"PatientName: {ds.get('PatientName', 'N/A')}")
print(f"SeriesDescription: {ds.get('SeriesDescription', 'N/A')}") # O tu tag de clasificación

print(f"RescaleIntercept: {ds.get('RescaleIntercept', 'N/A')}")
print(f"RescaleSlope: {ds.get('RescaleSlope', 'N/A')}")
print(f"RescaleType: {'Presente' if 'RescaleType' in ds else 'Ausente'}")

if 'ModalityLUTSequence' in ds and ds.ModalityLUTSequence:
    print("ModalityLUTSequence: Presente")
    lut_item = ds.ModalityLUTSequence[0]
    print(f"  LUTDescriptor: {lut_item.get('LUTDescriptor', 'N/A')}")
    print(f"  LUTExplanation: {lut_item.get('LUTExplanation', 'N/A')}")
    print(f"  ModalityLUTType: {lut_item.get('ModalityLUTType', 'N/A')}")
    print(f"  LUTData (longitud en bytes): {len(lut_item.get('LUTData', b''))}")
else:
    print("ModalityLUTSequence: Ausente o Vacía")

print(f"WindowCenter: {ds.get('WindowCenter', 'N/A')}")
print(f"WindowWidth: {ds.get('WindowWidth', 'N/A')}")
print(f"VOILUTSequence: {'Presente' if 'VOILUTSequence' in ds else 'Ausente'}")

print(f"PhotometricInterpretation: {ds.get('PhotometricInterpretation', 'N/A')}")
print(f"TransferSyntaxUID: {ds.file_meta.get('TransferSyntaxUID', 'N/A')}")

# Para verificar tags privados de linealización:
# Asume que config.PRIVATE_CREATOR_ID_LINEALIZATION y el grupo (ej. 0x00F1) son conocidos
private_creator_id = "TU_PRIVATE_CREATOR_ID_DE_CONFIG" # Reemplaza con el valor real
private_group = 0x00F1 # Reemplaza con el grupo real

# Encontrar el offset del bloque del creador
creator_offset = None
for i in range(0x10, 0xFF):
    tag = (private_group, i)
    if tag in ds and ds[tag].value == private_creator_id:
        creator_offset = i
        break

if creator_offset is not None:
    print(f"Bloque privado '{private_creator_id}' encontrado en offset del creador 0x{creator_offset:02X}")
    # block = ds.private_block(private_group, private_creator_id) # Obtener el bloque
    # rqa_tag_in_block = block.get(0x10) # Offset dentro del bloque
    # slope_tag_in_block = block.get(0x11) # Offset dentro del bloque

    # O acceder con el tag completo:
    tag_rqa = (private_group, (creator_offset << 8) | 0x10)
    tag_slope = (private_group, (creator_offset << 8) | 0x11)

    print(f"  RQA Type (Tag {tag_rqa!r}): {ds.get(tag_rqa, Dataset()).value}") # Dataset() como fallback para .value
    print(f"  Slope (Tag {tag_slope!r}): {ds.get(tag_slope, Dataset()).value}")
else:
    print(f"Bloque privado '{private_creator_id}' no encontrado.")