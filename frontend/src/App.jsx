import { Routes, Route, Navigate } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import Layout     from './components/Layout';
import Upload     from './pages/Upload';
import Profile    from './pages/Profile';
import Clean      from './pages/Clean';
import Transform  from './pages/Transform';
import EDA        from './pages/EDA';
import Stats      from './pages/Stats';
import NLP        from './pages/NLP';
import MLStrategy from './pages/MLStrategy';
import Report     from './pages/Report';
import AI         from './pages/AI';

export default function App() {
  return (
    <>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/upload" replace />} />
          <Route path="/upload"    element={<Upload />} />
          <Route path="/profile"   element={<Profile />} />
          <Route path="/clean"     element={<Clean />} />
          <Route path="/transform" element={<Transform />} />
          <Route path="/eda"       element={<EDA />} />
          <Route path="/stats"     element={<Stats />} />
          <Route path="/nlp"       element={<NLP />} />
          <Route path="/ml"        element={<MLStrategy />} />
          <Route path="/report"    element={<Report />} />
          <Route path="/ai"        element={<AI />} />
          <Route path="*"          element={<Navigate to="/upload" replace />} />
        </Route>
      </Routes>
      <ToastContainer position="bottom-right" autoClose={4000} newestOnTop theme="light" />
    </>
  );
}