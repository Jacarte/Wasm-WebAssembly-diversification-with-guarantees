from statistics import mean, median
import cv2
import numpy as np
from pdf2image import convert_from_path,pdfinfo_from_path
import sys
import os
import pytesseract
import json
from happytransformer import  HappyTextToText, TTSettings
from hashlib import md5
import math

import language_tool_python
tool = language_tool_python.LanguageTool('en-UK', config={ 'cacheSize': 1000, 'pipelineCaching': True })

model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=10000000)

OUT = os.path.abspath(os.path.dirname(__file__))


def get_edit_path(D, tokens_original, tokens_corrected):
    """
    Returns a list of tuples of (original, corrected) words
    """

    i = len(tokens_original)
    j = len(tokens_corrected)

    edit_path = []

    while (i > 0 or j > 0):
        if (i > 0 and j > 0 and tokens_original[i - 1] == tokens_corrected[j - 1]):
            edit_path.append((tokens_original[i - 1], i- 1, tokens_corrected[j - 1]))
            i = i - 1
            j = j - 1
        elif (i > 0 and D[i][j] == D[i - 1][j] + 1):
            edit_path.append((tokens_original[i - 1], i -1,  ""))
            i = i - 1
        elif (j > 0 and D[i][j] == D[i][j - 1] + 1):
            edit_path.append(("", -1, tokens_corrected[j - 1]))
            j = j - 1
        else:
            edit_path.append((tokens_original[i - 1], i-1, tokens_corrected[j - 1]))
            i = i - 1
            j = j - 1

    edit_path = [ e for e in edit_path if e[0] != e[2] ]
    return edit_path

def get_levensthein_edit_path(original, corrected):
    """
    Returns a list of tuples of (original, corrected) words
    """
    tokens_original = original.split()
    tokens_corrected = corrected.split()

    distances = np.zeros((len(tokens_original) + 1, len(tokens_corrected) + 1))

    a = 0
    b = 0
    c = 0

    for t1 in range(len(tokens_original) + 1):
        distances[t1][0] = t1

    for t2 in range(len(tokens_corrected) + 1):
        distances[0][t2] = t2

    for t1 in range(1, len(tokens_original) + 1):
        for t2 in range(1, len(tokens_corrected) + 1):
            if (tokens_original[t1-1] == tokens_corrected[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

            if (a <= b and a <= c):
                distances[t1][t2] = a + 1
            elif (b <= a and b <= c):
                distances[t1][t2] = b + 1
            else:
                distances[t1][t2] = c + 1

    D = distances[len(tokens_original)][len(tokens_corrected)]

    # Now get the edit path

    edit_path = get_edit_path(distances, tokens_original, tokens_corrected)
    
    return edit_path


def get_by_template(imagefile, templatefile, th=0.6):
    img_rgb = cv2.imread(imagefile)
    # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    template = cv2.imread(templatefile)

    blurred_img = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    blurred_template = cv2.GaussianBlur(template, (5, 5), 0)
    
    w, h, _ = template.shape[::-1]
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # Resize the image according to the scale
        resized = cv2.resize(blurred_img, (int(blurred_img.shape[1] * scale), int(blurred_img.shape[0] * scale)))
        
        # If the resized image is smaller than the template, break from the loop
        if resized.shape[0] < h or resized.shape[1] < w:
            break
        # Perform match operations in color
        res = cv2.matchTemplate(resized, blurred_template, cv2.TM_CCOEFF_NORMED)
        # Define a threshold for detection based on empirical testing
        threshold = th  #
        # Define a threshold for detection
        #threshold = 0.8
        
        # Find where the good matches are
        loc = np.where(res >= threshold)

        # Draw a rectangle around the matched region.
        rectangles = []
        for pt in zip(*loc[::-1]):
            # check if therre are not overlaping
            rectangles.append((pt[0]/scale, pt[1]/scale, w/scale, h/scale))
            cv2.rectangle(img_rgb, (int(pt[0] / scale), int(pt[1] / scale)), 
                      (int((pt[0] + w) / scale), int((pt[1] + h) / scale)), (0, 255, 255), 2)


    # Save res
    #cv2.imwrite(f"{imagefile}.res.jpg", res)
    # Save for debug
    cv2.imwrite(f"{imagefile}.template.jpg", img_rgb)
    return rectangles



def get_paragraphs_from_image(imagefile, sizemin=10000):
    print("Getting paragraphs from image")
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(imagefile)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,4))
    dilate = cv2.dilate(thresh, kernel, iterations=13)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    result = []

    filtered = []
    for c in cnts:

        x,y,w,h = cv2.boundingRect(c)

        if w*h >= sizemin: 
            filtered.append(c)
    
    non_overlap = []
    for c1 in range(len(filtered)):
        for c2 in range(len(filtered)):
            if c1 != c2:
                contour1 = filtered[c1]
                contour2 = filtered[c2]
                x,y,w,h = cv2.boundingRect(contour2)
                
                non_overlap.append(contour2)

                r1 = cv2.pointPolygonTest(contour1, (x, y), False) in [1, 0]
                r2 = cv2.pointPolygonTest(contour1, (x + w, y), False) in [1, 0]
                r3 = cv2.pointPolygonTest(contour1, (x, y + h), False) in [1, 0]
                r4 = cv2.pointPolygonTest(contour1, (x + w, y + h), False) in [1, 0]
                
                if all([r1, r2, r3, r4]):
                    # Remove contour2
                    non_overlap.pop()

    
    for c in non_overlap:
        # Remove those contours that are inside other

        # Get the largest and not overlaping
        # Use a segment tree, kd tree ?
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

        # Filter by size
        # print(w*h)
        if w*h >= sizemin: 
            result.append((
                image[y:y + h, x: x + w], w*h, (x, y, w, h))
            )
        # return each rectangle in an image

    # For debugging
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('dilate', dilate)
    # cv2.imshow('image', image)
    # cv2.waitKey()

    #cv2.imwrite(f"{imagefile}.contours.jpg", image)
    #cv2.imwrite(f"{imagefile}.dilate.jpg", dilate)
    return result

def get_score(text1, text2):


    ec1 = text1.encode('utf-8')
    ec2 = text2.encode('utf-8')

    if len(ec1) < len(ec2):
        ec1, ec2 = ec2, ec1

    DTW = [[10000000000 for _ in range(len(ec1) + 1) ] for _ in range(len(ec2) + 1)]

    def d(c1, c2):
        if c1 == c2:
            return 0
        else:
            return 300

    DTW[0][0] = 0
    for i in range(len(ec2) + 1):
        for j in range(len(ec1) + 1):
            if i == 0:
                DTW[i][j] = 100*j
                continue
            if j == 0:
                DTW[i][j] = 100*i
                continue
            
            c1 = ec2[i - 1]
            c2 = ec1[j - 1]

            DTW[i][j] = min(
                DTW[i - 1][j - 1] + d(c1, c2),
                min(DTW[i - 1][j] + 100, DTW[i][j - 1] + 100)
            )
    #print(text1, text2, DTW[i][j])
    return DTW[i][j]

def get_terms(text):

    REMOVE = ['i.e.']
    TO_SPLIT = [" ", "\n"]
    
    tokens = [text]
    for t in TO_SPLIT:
        tmp = []
        for token in tokens:
            tokensn = token.split(t)
            tmp += tokensn
        tokens = tmp
    
    tokens = list(filter(lambda x: x not in REMOVE, tokens))
    
    # 2-grams
    _2grams = zip(tokens, tokens[1:])
    _2grams = list(_2grams)

    # 3-grams
    _3grams = zip(_2grams, tokens[2:])
    _3grams = list(_3grams)    
    print(_3grams)
    return []
    
def ai_spell_check(ID, text, imagedata,pagen, tesseractdata, rect, relative, words2ignore = []):

    # TODO split the text by lines
    lines = text.split("\n")

    suggested = ""
    for l in lines:

        grammar = f"grammar: {l}"
        happy_suggestions = model.generate_text(grammar, beam_settings)
        happy_suggestions = happy_suggestions.text
       
        suggested += happy_suggestions + "\n"

    edit_distance = get_levensthein_edit_path(text, suggested)
    print(edit_distance)
    print("=====================================")
    
    suggested_tokens = suggested.split(" ")
    suggested_tokens = enumerate(suggested_tokens)
    suggested = " ".join([f"({i}){s}" for i, s in suggested_tokens])
    print(suggested)
    print("-------------------------------------")
    text_tokens = text.split(" ")
    text_tokens = enumerate(text_tokens)
    text = " ".join([f"({i}){s}" for i, s in text_tokens])
    print(text)
    print("=====================================")
    boxes = len(tesseractdata['level'])
    obs = []
    hashes = set()

    if len(edit_distance) > 0:

        for original, index, corrected in edit_distance:
            obj = dict(
                    # Unique id
                    matches = []
                )
        
            tpe="replace"
            category = "ai-replace"
            replacements = [corrected]
            if corrected == "":
                tpe = "delete"
                category = "ai-remove"
                replacements = []
            if original == "":
                tpe = "insert"
                category = "ai-insert"
                replacements = [corrected]

            chunk = original
            if chunk.lower() not in words2ignore:
                obj['matches'].append(dict(
                    message=f"{tpe} '{original}' by '{corrected}'",
                    replacements=replacements,
                    ruleId="AI",
                    offsetInContext=text.index(original),
                    category=category,
                    offset=0,
                    errorLength=len(original),
                    text=text,
                    happy=suggested
                ))
                obj['places'] = []
                globalx, globaly, _, _ = rect
                margin = 3
                # check each box and collect the text again, if it is equal to the error chunk, then, that is the place
                scores = []
                for i in range(boxes):
                    texti = tesseractdata['text'][i]
                    score = get_score(texti, chunk)
                    scores.append((score, i, index, tesseractdata['width'][i], tesseractdata['height'][i]))


                # the smaller the rectangle the better
                scores = sorted(scores, key=lambda x: x[2]*x[3])
                # this should solve the location
                scores = sorted(scores, key=lambda x: abs(x[1] - x[2]))
                scores = sorted(scores, key=lambda x: x[0])
                # then sort by position top, left

                # Draw all rectangles with the same score
                for sc, i, index, w, h in scores:
                    x, y, w, h = tesseractdata['left'][i], tesseractdata['top'][i], tesseractdata['width'][i], tesseractdata['height'][i]
                    if h*w == 0 or min(w, h)/max(w, h) < 1/6:
                        continue
                    cv2.rectangle(imagedata, (x + globalx - margin, y + globaly - margin), (x + w  + globalx + margin, y + h + globaly + margin), (255,36,12), 2)

                    if len(obj['places']) == 0:
                        # But only keep one ?
                        obj['places'].append(dict(
                            x=x + globalx ,
                            y =  y + globaly,
                            w = w,
                            h = h
                        ))
                obj['pagefile'] = relative
                obj['chunk'] = chunk
                obj['pageannotatedfile'] = f"rois/annotated_{pagen}.png"
                # Add a different color per rule and annotate the image

                hashi = md5(json.dumps(obj).encode('utf-8')).hexdigest()
                if hashi in hashes:
                    continue
                
                obj['id'] = ID
                ID += 1
                obs.append(obj)
                hashes.add(hashi)
    
    return obs, ID


def spell_check(ID, text, imagedata,pagen, tesseractdata, rect, relative, words2ignore = []):
    matches = tool.check(text)
    # Check missing references in imagedata

    boxes = len(tesseractdata['level'])
    obs = []
    if len(matches) > 0:
        
        obj = dict(
                    # Unique id
                    id=ID,
                    matches = []
                    )

        for m in matches:
            chunk = text[m.offset:m.offset + m.errorLength]

            if not chunk:
                continue
            if chunk.lower() not in words2ignore:
                obj['matches'].append(dict(
                    message=m.message,
                    replacements=m.replacements,
                    ruleId=m.ruleId,
                    offsetInContext=m.offsetInContext,
                    category=m.category,
                    offset=m.offset,
                    errorLength=m.errorLength,
                    text=text
                ))
                obj['places'] = []
                #print(m.offsetInContext, m.offset)
                print(text)
                print(m)
                print()
                globalx, globaly, _, _ = rect
                margin = 3
                # check each box and collect the text again, if it is equal to the error chunk, then, that is the place
                scores = []
                for i in range(boxes):
                    text = tesseractdata['text'][i]
                    score = get_score(text, chunk)
                    scores.append((score, i, tesseractdata['width'][i], tesseractdata['height'][i]))


                # the smaller the rectangle the better
                scores = sorted(scores, key=lambda x: x[2]*x[3])
                scores = sorted(scores, key=lambda x: x[0])


                # Draw all rectangles with the same score
                for sc, i, w, h in scores:
                    if w < 80:
                        continue
                    if h < 10:
                        continue
                    if sc != scores[0][0]:
                        break
                    x, y, w, h = tesseractdata['left'][i], tesseractdata['top'][i], tesseractdata['width'][i], tesseractdata['height'][i]
                    cv2.rectangle(imagedata, (x + globalx - margin, y + globaly - margin), (x + w  + globalx + margin, y + h + globaly + margin), (255,36,12), 2)

                    if len(obj['places']) == 0:
                        # But only keep one ?
                        obj['places'].append(dict(
                            x=x + globalx ,
                            y =  y + globaly,
                            w = w,
                            h = h
                        ))
                obj['pagefile'] = relative
                obj['chunk'] = chunk
                obj['pageannotatedfile'] = f"rois/annotated_{pagen}.png"
                # Add a different color per rule and annotate the image

                ID += 1
                obs.append(obj)
    return obs, ID

def process_pdf(pdffile, ignore):
    words2ignore = open(ignore, 'r').readlines()
    words2ignore = [l.lower().strip().replace("\n", "").replace("\\\\", "\\") for l in words2ignore]
    info = pdfinfo_from_path(pdffile, userpw=None, poppler_path=None)
    maxPages = info["Pages"]

    print("Pages", maxPages)
    pagen = 0
    REPORT = {}
    COUNTS = {

    }
    ID = 0

    STEP=5
    DPI=300
    for page in range(1, maxPages+1, STEP):
        print("Processing pages", page, min(page + STEP - 1, maxPages))
        images = convert_from_path(pdffile, dpi=DPI, first_page=page, last_page=min(page + STEP - 1, maxPages))

        TEXT_CACHE = set()
        for image in images:
            try:
                # Save temp
                REPORTPAGE = []

                relative = f"rois/page_{pagen}.jpg"
                name = f"{OUT}/{relative}"
                image.save(name, "JPEG")

                squares = get_paragraphs_from_image(name)


                i = 0
                imagedata = cv2.imread(name)

                # Detect missing references
                ROI_COUNT = 0
                for roi, s, rect in squares:
                    # Comment this, it is debugging
                    data = pytesseract.image_to_data(roi,output_type='dict')#, config=custom_config)
                    #print(data)
                    boxes = len(data['level'])
                    text = ""
                    for i in range(boxes):
                        text += " " +  data['text'][i]
                    
                        
                    cv2.imwrite(f"{OUT}/rois/roi_{i}_{pagen}_{s}.png", roi)
                    # Some sanitization
                    text = text.replace("\n", " ")
                    text = text.replace(". ", ".\n")
                    
                    while True:
                        old = text
                        text = text.replace("  ", " ")
                        if old == text:
                            break

                    text = text.replace(" .", ".")
                    text = text.replace("- ", "-")
                    text = text.replace(" )", ")")
                    text = text.replace(" ]", "]")
                    text = text.replace(" }", "}")
                    text = text.replace("( ", "(")
                    text = text.replace("https: ", "https:")
                    text = text.strip()

                    if text in TEXT_CACHE:
                        print("Text already in cache")
                        continue

                    TEXT_CACHE.add(text)
                    
                        
                    # Call language tool or any other
                    
                    # Sentiment analysis?
                    # scores = get_score(text)
                    # call happy model here
                    
                    # This does spell check

                    obs_ai, IO = ai_spell_check(ID, text, imagedata, pagen, data, rect, relative, words2ignore)
                    obs, _ = spell_check(IO, text, imagedata, pagen, data, rect, relative, words2ignore)
                    # here do different things
                    REPORTPAGE += obs_ai
                    REPORTPAGE += obs
                    print("Report count", len(REPORTPAGE))
                    ROI_COUNT += 1


                if len(REPORTPAGE) > 0:
                    cv2.imwrite(f"{OUT}/rois/annotated_{pagen}.png", imagedata)

                    if name not in REPORT:
                        REPORT[relative] = []

                    REPORT[relative] += REPORTPAGE
                else:
                    os.remove(name)

                i += 1
                pagen += 1
            except KeyboardInterrupt:
                break

    REPORT["counts"] = COUNTS
    open(f"{OUT}/report.json", "w").write(json.dumps(REPORT, indent=4))

if __name__ == '__main__':
    process_pdf(sys.argv[1], sys.argv[2])
