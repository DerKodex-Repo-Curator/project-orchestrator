#!/usr/bin/env python3
"""Collect raw results. No scoring, no claim that graph-edge counts equal call sites."""
import base64,json,subprocess,time,urllib.error,urllib.parse,urllib.request
from pathlib import Path
BASE=Path(__file__).resolve().parent
OUT=BASE/'results'
SYMBOLS=['with_cognitive_routing','set_inference_engine','build_enrichment_pipeline','close_decision_with']

def http(url,body=None,headers=None):
    req=urllib.request.Request(url,data=None if body is None else json.dumps(body).encode(),
                               headers={'Content-Type':'application/json',**(headers or {})})
    with urllib.request.urlopen(req,timeout=1800) as r:
        result=json.loads(r.read())
        if isinstance(result,dict) and isinstance(result.get('errors'),list) and result['errors']:
            raise RuntimeError(json.dumps(result['errors']))
        return result

def command(service,args):
    p=subprocess.run(['docker','compose','exec','-T',service,*args],cwd=BASE,capture_output=True,text=True,timeout=1800)
    if p.returncode: raise RuntimeError(p.stderr[-2000:] or p.stdout[-2000:])
    try:
        result=json.loads(p.stdout)
        if isinstance(result,dict) and result.get('isError'):
            raise RuntimeError(json.dumps(result))
        return result
    except json.JSONDecodeError: return {'raw':p.stdout}

def measure(name,fn,report):
    start=time.perf_counter()
    try:
        result=fn(); entry={'status':'ok','wall_seconds':time.perf_counter()-start,
                           'response_bytes':len(json.dumps(result,ensure_ascii=False).encode()),'result':result}
    except Exception as e:
        entry={'status':'error','wall_seconds':time.perf_counter()-start,'error':str(e)}
    (OUT/(name+'.json')).write_text(json.dumps(entry,ensure_ascii=False,indent=2))
    report[name]={k:v for k,v in entry.items() if k!='result'}
    print(name,entry['status'],round(entry['wall_seconds'],3),'seconds',flush=True)
    return entry

def main():
    OUT.mkdir(exist_ok=True)
    report={'method':'One sequential run. Timings include HTTP/CLI transport and process startup; not a statistical benchmark.',
            'count_semantics':'PO CALLS edges, CBM graph edges, and reader syntactic call sites are different quantities.',
            'token_measurement':'UTF-8 response byte counts only; no tokenizer-exact token count.'}
    report['compose_files']=__import__('os').environ.get('COMPOSE_FILE','compose.yaml')
    api='http://127.0.0.1:18080'
    try:
        project=http(api+'/api/projects/po-self')
    except urllib.error.HTTPError as e:
        if e.code!=404: raise
        project=http(api+'/api/projects',{'name':'PO self-index comparison','slug':'po-self','root_path':'/workspace/po'})
    report['po_files_before_sync']=project.get('file_count')
    report['po_warm_state']=bool(project.get('last_synced'))
    sync=measure('po-index',lambda:http(api+'/api/projects/po-self/sync?force=true',{}),report)
    if sync['status']=='ok' and sync['result'].get('errors',0):
        report['po_index_has_errors']=True
    measure('po-project',lambda:http(api+'/api/projects/po-self'),report)
    measure('cbm-before',lambda:command('cbm',['codebase-memory-mcp','cli','--json','list_projects']),report)
    measure('cbm-index',lambda:command('cbm',['codebase-memory-mcp','cli','--json','index_repository','--repo-path','/workspace/po','--name','po','--mode','full']),report)
    measure('cbm-projects',lambda:command('cbm',['codebase-memory-mcp','cli','--json','list_projects']),report)
    measure('reader-index',lambda:command('reader',['python3','/opt/reader/code_reader.py','index',json.dumps({'project':'po','root':'/workspace/po'})]),report)
    # Meilisearch index writes are asynchronous. Wait for its outstanding tasks.
    env=dict(line.split('=',1) for line in (BASE/'.env').read_text().splitlines() if '=' in line)
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        tasks=command('po',['curl','-fsS','-H','Authorization: Bearer '+env['MEILI_MASTER_KEY'],
                            'http://meilisearch:7700/tasks?statuses=enqueued,processing&limit=1'])
        if tasks.get('total',0)==0: break
        time.sleep(1)
    else:
        report['meilisearch_drain_timeout']=True
    for symbol in SYMBOLS:
        q=urllib.parse.urlencode({'function':symbol,'direction':'callers','depth':1,'project_slug':'po-self'})
        measure('po-callers-'+symbol,lambda q=q:http(api+'/api/code/callgraph?'+q),report)
        q=urllib.parse.urlencode({'query':symbol,'project_slug':'po-self','limit':5})
        measure('po-search-'+symbol,lambda q=q:http(api+'/api/code/search?'+q),report)
        measure('cbm-search-'+symbol,lambda symbol=symbol:command('cbm',['codebase-memory-mcp','cli','--json','search_graph','--project','po','--name-pattern','^'+symbol+'$','--label','Function']),report)
        measure('cbm-callers-'+symbol,lambda symbol=symbol:command('cbm',['codebase-memory-mcp','cli','--json','trace_path','--project','po','--function-name',symbol,'--direction','inbound']),report)
        measure('reader-callers-'+symbol,lambda symbol=symbol:command('reader',['python3','/opt/reader/code_reader.py','calls',json.dumps({'project':'po','name':symbol,'limit':100,'include_tests':True})]),report)
    headers={'Authorization':'Basic '+base64.b64encode(('neo4j:'+env['NEO4J_PASSWORD']).encode()).decode()}
    pid=project['id']
    statements=[
        {'statement':'MATCH (f:File {project_id:$pid}) RETURN count(f) AS files','parameters':{'pid':pid}},
        {'statement':'MATCH (f:File {project_id:$pid})-[:CONTAINS]->(s) RETURN labels(s) AS labels,count(s) AS count','parameters':{'pid':pid}},
        {'statement':'MATCH (f:File {project_id:$pid})-[:CONTAINS]->(:Function)-[r:CALLS]->() RETURN count(r) AS calls','parameters':{'pid':pid}}]
    measure('po-graph-counts',lambda:http('http://127.0.0.1:17474/db/neo4j/tx/commit',{'statements':statements},headers),report)
    (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('Raw results saved under results/. No automated winner is inferred.')
if __name__=='__main__': main()
