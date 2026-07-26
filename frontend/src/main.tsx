import React, {FormEvent, useEffect, useState} from "react";
import {createRoot} from "react-dom/client";
import {
  Alert, AppBar, Box, Button, Card, CardContent, Chip, CircularProgress,
  Container, CssBaseline, Dialog, DialogContent, DialogTitle, LinearProgress,
  List, ListItem, ListItemText, Paper, Stack, Tab, Tabs, TextField, ThemeProvider,
  Toolbar, Typography, createTheme,
} from "@mui/material";
import CloudUploadOutlined from "@mui/icons-material/CloudUploadOutlined";
import SendRounded from "@mui/icons-material/SendRounded";
import "./styles.css";

const auth = {"X-Tenant-ID":"demo", "X-User-ID":"demo", "X-Groups":"research"};
const jsonHeaders = {...auth, "Content-Type":"application/json"};
type Message = {role:"user"|"assistant", text:string, citations?:string[]};
type DocumentItem = {id:string, title:string, created_at:string, index_status:string};
type Citation = {document_title:string, excerpt:string, page?:number, section?:string};

const theme = createTheme({
  palette: {mode:"light", primary:{main:"#315c49"}, secondary:{main:"#c45f35"}, background:{default:"#f5f3ed"}},
  shape: {borderRadius: 12},
  typography: {fontFamily:'Inter, Roboto, system-ui, sans-serif', h3:{fontWeight:700}},
});

async function checked(response: Response) {
  if (response.ok) return response;
  const body = await response.json().catch(() => ({}));
  throw new Error(body.error?.message || `Request failed (${response.status})`);
}

function App() {
  const [tab, setTab] = useState(0);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [session, setSession] = useState<string>();
  const [query, setQuery] = useState("");
  const [stage, setStage] = useState("ready");
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingText, setStreamingText] = useState("");
  const [lastQuery, setLastQuery] = useState("");
  const [citation, setCitation] = useState<Citation>();
  const [error, setError] = useState("");

  const refreshDocuments = () => fetch("/api/v1/documents", {headers:auth}).then(checked).then(r=>r.json()).then(setDocuments).catch(e=>setError(e.message));
  useEffect(()=>{void refreshDocuments();}, []);

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); const formElement=event.currentTarget; setError(""); setUploading(true); setProgress(2);
    try {
      const form = new FormData(formElement);
      const accepted = await fetch("/api/v1/documents", {method:"POST", headers:auth, body:form}).then(checked).then(r=>r.json());
      for (;;) {
        const job = await fetch(`/api/v1/ingestion-jobs/${accepted.ingestion_job.id}`, {headers:auth}).then(checked).then(r=>r.json());
        setProgress(job.progress);
        if (job.status === "complete") break;
        if (job.status === "failed") throw new Error(job.error || "Ingestion failed");
        await new Promise(resolve => setTimeout(resolve, 750));
      }
      formElement.reset(); await refreshDocuments();
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Upload failed"); }
    finally { setUploading(false); }
  }

  async function ask(event: FormEvent) {
    event.preventDefault(); if (!query.trim()) return; setError("");
    try {
      const id = session || await fetch("/api/v1/chat/sessions", {method:"POST", headers:jsonHeaders, body:"{}"}).then(checked).then(r=>r.json()).then(r=>{setSession(r.id); return r.id});
      const text=query; setLastQuery(text); setQuery(""); setStreamingText(""); setMessages(items=>[...items,{role:"user",text}]); setStage("planning");
      const run = await fetch(`/api/v1/chat/sessions/${id}/messages`, {method:"POST",headers:jsonHeaders,body:JSON.stringify({content:text})}).then(checked).then(r=>r.json());
      const response = await fetch(run.events_url,{headers:auth}).then(checked);
      const reader=response.body!.getReader(), decoder=new TextDecoder(); let buffer="";
      for (;;) {
        const {done,value}=await reader.read(); if(done) break; buffer+=decoder.decode(value,{stream:true});
        const frames=buffer.split("\n\n"); buffer=frames.pop()!;
        for(const frame of frames){
          const line=frame.split("\n").find(item=>item.startsWith("data: ")); if(!line) continue;
          const data=JSON.parse(line.slice(6));
          if(data.type==="status") setStage(data.stage);
          if(data.type==="token") setStreamingText(value=>value+data.content);
          if(data.type==="answer") {setStreamingText(""); setMessages(items=>[...items,{role:"assistant",text:data.content,citations:data.citations}]);}
          if(data.type==="complete") setStage("ready");
          if(data.type==="error") throw new Error(data.message);
        }
      }
    } catch(reason) {setStage("ready"); setError(reason instanceof Error?reason.message:"Chat failed");}
  }

  async function openCitation(id:string) {
    try { setCitation(await fetch(`/api/v1/citations/${id}`,{headers:auth}).then(checked).then(r=>r.json())); }
    catch(reason) { setError(reason instanceof Error?reason.message:"Citation unavailable"); }
  }

  return <ThemeProvider theme={theme}><CssBaseline/>
    <AppBar position="static" elevation={0}><Toolbar><Typography variant="h6" sx={{flexGrow:1}}>Connected Evidence</Typography><Chip label={stage} color="secondary" size="small"/></Toolbar></AppBar>
    <Container maxWidth="lg" sx={{py:4}}>
      <Typography variant="overline" color="primary">ENTERPRISE AGENTIC RAG</Typography>
      <Typography variant="h3" gutterBottom>Research with traceable answers.</Typography>
      <Tabs value={tab} onChange={(_,value)=>setTab(value)} aria-label="Workspace"><Tab label="Documents"/><Tab label="Chat"/></Tabs>
      {error && <Alert severity="error" onClose={()=>setError("")} action={lastQuery?<Button color="inherit" onClick={()=>{setQuery(lastQuery);setError("")}}>Retry</Button>:undefined} sx={{my:2}}>{error}</Alert>}
      {tab===0 && <Stack spacing={3} sx={{mt:3}}>
        <Paper component="form" onSubmit={upload} sx={{p:3}}><Stack spacing={2}>
          <Typography variant="h6">Ingest a document</Typography>
          <TextField name="title" label="Document title" required/>
          <TextField name="acl_groups" label="ACL groups" defaultValue="research" helperText="Comma-separated groups"/>
          <Button component="label" variant="outlined" startIcon={<CloudUploadOutlined/>}>Choose PDF or text<input name="file" type="file" accept=".pdf,.txt,.md" required hidden/></Button>
          {uploading && <LinearProgress variant="determinate" value={progress}/>} 
          <Button type="submit" variant="contained" disabled={uploading}>{uploading?<CircularProgress size={22}/>:"Upload and index"}</Button>
        </Stack></Paper>
        <Card><CardContent><Typography variant="h6">Authorized documents</Typography><List>{documents.length?documents.map(item=><ListItem key={item.id} divider><ListItemText primary={item.title} secondary={new Date(item.created_at).toLocaleString()}/><Chip label={item.index_status} color={item.index_status==="active"?"success":"default"} size="small"/></ListItem>):<ListItem><ListItemText primary="No documents yet" secondary="Upload the transcript to begin."/></ListItem>}</List></CardContent></Card>
      </Stack>}
      {tab===1 && <Box sx={{mt:3}}>
        <Paper className="messages" aria-live="polite">{messages.length===0&&!streamingText?<Stack sx={{height:"100%",alignItems:"center",justifyContent:"center"}}><Typography variant="h5">Ask across your indexed evidence</Typography><Typography color="text.secondary">Answers refuse unsupported claims and link back to sources.</Typography></Stack>:<>{messages.map((message,index)=><Box key={index} className={`message ${message.role}`}><Typography variant="caption" sx={{fontWeight:700}}>{message.role==="user"?"You":"Research agent"}</Typography><Typography sx={{whiteSpace:"pre-wrap"}}>{message.text}</Typography>{message.citations?.map((id,n)=><Button key={id} size="small" onClick={()=>openCitation(id)}>Source {n+1}</Button>)}</Box>)}{streamingText&&<Box className="message assistant"><Typography variant="caption" sx={{fontWeight:700}}>Research agent · {stage}</Typography><Typography sx={{whiteSpace:"pre-wrap"}}>{streamingText}</Typography></Box>}</>}</Paper>
        <Paper component="form" onSubmit={ask} sx={{p:2,mt:2,display:"flex",gap:1}}><TextField fullWidth multiline maxRows={5} label="Question" value={query} onChange={event=>setQuery(event.target.value)}/><Button type="submit" variant="contained" endIcon={<SendRounded/>}>Ask</Button></Paper>
      </Box>}
    </Container>
    <Dialog open={Boolean(citation)} onClose={()=>setCitation(undefined)} maxWidth="sm" fullWidth><DialogTitle>{citation?.document_title}</DialogTitle><DialogContent><Typography color="text.secondary" gutterBottom>{citation?.section}{citation?.page?` · page ${citation.page}`:""}</Typography><Typography>{citation?.excerpt}</Typography></DialogContent></Dialog>
  </ThemeProvider>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App/></React.StrictMode>);
