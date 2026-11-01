"""Small opt-in live retrieval controls for Jev 1.13 and Liquid d1-ultra.

This checks real API wiring/ranking/usage, not benchmark intelligence or calibration.
Only synthetic public examples are sent. Default budget: $0.002 per model, four calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

CASES = [
    ('python','Fix a Python CSV writer: preserve the original file if serialization fails.',[
        ('a','Stage CSV output in a temporary file in the same directory; after successful serialization use os.replace; clean up on failure.'),
        ('b','Adjust CSS border radius and hover colors for a toolbar.'),
        ('c','Use exact rational arithmetic for a finite mathematical sum.')],'a'),
    ('mathematics','Вычислить точно сумму рациональных слагаемых и получить несократимую дробь.',[
        ('a','Настройка Docker Compose: именованные тома и сети сервисов.'),
        ('b','Для точной рациональной суммы применить fractions.Fraction либо sympy.Rational без float.'),
        ('c','Проверка ABI C++ и время жизни указателей в библиотеке.')],'b'),
    ('algorithms','Find a counterexample where greedy coin change uses too many coins.',[
        ('a','Use aria-label and keyboard focus for accessible UI buttons.'),
        ('b','Safely rotate application logs without losing file permissions.'),
        ('c','Enumerate small amounts; compare largest-first coin selection against an independent optimal search.')],'c'),
    ('linux','Найти строки с определённым идентификатором в исходниках Linux-проекта без изменения файлов.',[
        ('a','Использовать rg -n -F -- ИДЕНТИФИКАТОР путь/к/исходникам; затем прочитать нужный диапазон строк.'),
        ('b','Разложить рациональную функцию на простые дроби и вывести формулу суммы.'),
        ('c','Сравнить раскраски графа и доказать нижнюю границу хроматического числа.')],'a'),
]


async def run(args):
    from src.coding.navigation import DecisionsScorer
    report={'scenario':'decision_retrieval_integration_not_benchmark','models':[]}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    for name in args.models:
        scorer=DecisionsScorer({'model':name,'max_calls':len(CASES),'max_cost_usd':.002})
        rows=[]
        for case,query,items,expected in CASES:
            started=time.monotonic()
            try:
                candidates=[{'id':cid,'text':text} for cid,text in items]
                scores=await scorer.score(query,candidates)
                cached=await scorer.score(query,candidates)
                best=max(scores,key=scores.get)
                passed=best==expected and all(scores[expected]>score for cid,score in scores.items() if cid!=expected)
                row={'case':case,'scores':scores,'expected':expected,'passed':passed,'cache_same':cached==scores}
            except Exception as exc:
                row={'case':case,'passed':False,'error':type(exc).__name__+': '+str(exc)[:300]}
            row['seconds']=round(time.monotonic()-started,3)
            rows.append(row)
            print(json.dumps({'model':name,**row},ensure_ascii=False),flush=True)
        entry={'model':name,'rows':rows,'usage':scorer.usage()}
        report['models'].append(entry)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps({'model':name,'usage':scorer.usage()},ensure_ascii=False),flush=True)
    return 0 if all(r['passed'] for m in report['models'] for r in m['rows']) else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--models',nargs='+',default=['typesafe/jev-1.13','liquid/d1-ultra'])
    parser.add_argument('--output',default='data/eval/decision-models-scenario.json')
    args=parser.parse_args()
    if args.live: raise SystemExit(asyncio.run(run(args)))
    print('No API calls. Add --live for bounded cloud-only checks.')
