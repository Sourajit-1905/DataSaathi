import axios from 'axios';
import { toast } from 'react-toastify';

const client = axios.create({ baseURL: '/api', timeout: 120_000 });

client.interceptors.response.use(
    (res) => res.data,
    (err) => {
        if (err.code === 'ECONNABORTED' || err.code === 'ERR_NETWORK') {
            toast.error('Server not responding — is the backend running on port 8000?');
        }
        const data = err.response?.data ?? {};
        return Promise.reject({
            status: err.response?.status,
            error: data.error ?? data.detail ?? 'Something went wrong',
            detail: data.detail ?? '',
            code: data.code ?? err.response?.status,
        });
    }
);

export const datasetAPI = {
    upload: (file) => {
        const fd = new FormData();
        fd.append('file', file);
        return client.post('/dataset/upload', fd, { headers: { 'Content-Type': 'multipart/form-data' } });
    },
    preview: (sid) => client.get(`/dataset/${sid}/preview`),
    info: (sid) => client.get(`/dataset/${sid}/info`),
    reset: (sid) => client.post(`/dataset/${sid}/reset`),
};

export const profileAPI = {
    run: (sid) => client.post(`/profile/${sid}/run`),
    getResult: (sid) => client.get(`/profile/${sid}/result`),
};

export const cleanAPI = {
    recommend: (sid, issueType) => client.get(`/clean/${sid}/recommend/${issueType}`),
    preview: (sid, body) => client.post(`/clean/${sid}/preview`, body),
    apply: (sid, body) => client.post(`/clean/${sid}/apply`, body),
    undo: (sid) => client.post(`/clean/${sid}/undo`),
    history: (sid) => client.get(`/clean/${sid}/history`),
};

export const transformAPI = {
    recommend: (sid) => client.get(`/transform/${sid}/recommend`),
    preview: (sid, body) => client.post(`/transform/${sid}/preview`, body),
    apply: (sid, body) => client.post(`/transform/${sid}/apply`, body),
    undo: (sid) => client.post(`/transform/${sid}/undo`),
    history: (sid) => client.get(`/transform/${sid}/history`),
};

export const edaAPI = {
    analyze: (sid, body) => client.post(`/eda/${sid}/analyze`, body),
    question: (sid, body) => client.post(`/eda/${sid}/question`, body),
    insights: (sid) => client.get(`/eda/${sid}/insights`),
    history: (sid) => client.get(`/eda/${sid}/history`),
};

export const statsAPI = {
    recommend: (sid, body) => client.post(`/stats/${sid}/recommend`, body),
    run: (sid, body) => client.post(`/stats/${sid}/run`, body),
    history: (sid) => client.get(`/stats/${sid}/history`),
};

export const nlpAPI = {
    detectColumns: (sid) => client.get(`/nlp/${sid}/detect-columns`),
    run: (sid, body) => client.post(`/nlp/${sid}/run`, body),
    question: (sid, body) => client.post(`/nlp/${sid}/question`, body),
    insights: (sid) => client.get(`/nlp/${sid}/insights`),
    history: (sid) => client.get(`/nlp/${sid}/history`),
};

export const mlAPI = {
    recommend: (sid, body) => client.post(`/ml/${sid}/recommend`, body),
    getResult: (sid) => client.get(`/ml/${sid}/result`),
};

export const reportAPI = {
    generate: (sid, body) => client.post(`/report/${sid}/generate`, body),
    download: (sid, format) => `/api/report/${sid}/download/${format}`,
    history: (sid) => client.get(`/report/${sid}/history`),
};

export const aiAPI = {
    chat: (sid, message) => client.post(`/ai/${sid}/chat`, { message }),
    history: (sid) => client.get(`/ai/${sid}/history`),
    clear: (sid) => client.post(`/ai/${sid}/clear`),
    insights: (sid, refresh) => client.get(`/ai/${sid}/insights`, { params: { refresh } }),
    explain: (sid, body) => client.post(`/ai/${sid}/explain`, body),
    summarize: (sid) => client.get(`/ai/${sid}/summarize`),
    suggestions: (sid) => client.get(`/ai/${sid}/suggestions`),
    status: () => client.get('/ai/status'),
};