/*
 * Interface wrapper code.
 *
 * Generated by SIP 4.19.3
 *
 * Copyright (C) 2009 Autodesk, Inc. and/or its licensors.
 * All Rights Reserved.
 * 
 * The coded instructions, statements, computer programs, and/or related material 
 * (collectively the "Data") in these files contain unpublished information 
 * proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 * Canada and United States of America federal copyright law and by international 
 * treaties. 
 * 
 * The Data may not be disclosed or distributed to third parties, in whole or in
 * part, without the prior written consent of Autodesk, Inc. ("Autodesk").
 * 
 * THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 * ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 * WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 * BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 * NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 * WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 * OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 * 
 * IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 * OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 * OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 * SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 * OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 * HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 * FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 * ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 * WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 * OR DAMAGE. 
 */

#include "sipAPIfbx.h"




PyDoc_STRVAR(doc_FbxLayer_GetNormals, "GetNormals(self) -> FbxLayerElementNormal");

extern "C" {static PyObject *meth_FbxLayer_GetNormals(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetNormals(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
             ::FbxLayerElementNormal*sipRes;

            sipRes = sipCpp->GetNormals();

            return sipConvertFromType(sipRes,sipType_FbxLayerElementNormal,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetNormals, doc_FbxLayer_GetNormals);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetTangents, "GetTangents(self) -> FbxLayerElementTangent");

extern "C" {static PyObject *meth_FbxLayer_GetTangents(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetTangents(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
             ::FbxLayerElementTangent*sipRes;

            sipRes = sipCpp->GetTangents();

            return sipConvertFromType(sipRes,sipType_FbxLayerElementTangent,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetTangents, doc_FbxLayer_GetTangents);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetBinormals, "GetBinormals(self) -> FbxLayerElementBinormal");

extern "C" {static PyObject *meth_FbxLayer_GetBinormals(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetBinormals(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
             ::FbxLayerElementBinormal*sipRes;

            sipRes = sipCpp->GetBinormals();

            return sipConvertFromType(sipRes,sipType_FbxLayerElementBinormal,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetBinormals, doc_FbxLayer_GetBinormals);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetMaterials, "GetMaterials(self) -> FbxLayerElementMaterial");

extern "C" {static PyObject *meth_FbxLayer_GetMaterials(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetMaterials(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
             ::FbxLayerElementMaterial*sipRes;

            sipRes = sipCpp->GetMaterials();

            return sipConvertFromType(sipRes,sipType_FbxLayerElementMaterial,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetMaterials, doc_FbxLayer_GetMaterials);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetPolygonGroups, "GetPolygonGroups(self) -> FbxLayerElementPolygonGroup");

extern "C" {static PyObject *meth_FbxLayer_GetPolygonGroups(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetPolygonGroups(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
             ::FbxLayerElementPolygonGroup*sipRes;

            sipRes = sipCpp->GetPolygonGroups();

            return sipConvertFromType(sipRes,sipType_FbxLayerElementPolygonGroup,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetPolygonGroups, doc_FbxLayer_GetPolygonGroups);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetUVs, "GetUVs(self, FbxLayerElement.EType = FbxLayerElement.eTextureDiffuse) -> FbxLayerElementUV");

extern "C" {static PyObject *meth_FbxLayer_GetUVs(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetUVs(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElement::EType a0 = FbxLayerElement::eTextureDiffuse;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B|E", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElement_EType, &a0))
        {
             ::FbxLayerElementUV*sipRes;

            sipRes = sipCpp->GetUVs(a0);

            return sipConvertFromType(sipRes,sipType_FbxLayerElementUV,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetUVs, doc_FbxLayer_GetUVs);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetUVSetCount, "GetUVSetCount(self) -> int");

extern "C" {static PyObject *meth_FbxLayer_GetUVSetCount(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetUVSetCount(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
            int sipRes;

            sipRes = sipCpp->GetUVSetCount();

            return SIPLong_FromLong(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetUVSetCount, doc_FbxLayer_GetUVSetCount);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetUVSetChannels, "GetUVSetChannels(self) -> FbxLayerElementTypeArray");

extern "C" {static PyObject *meth_FbxLayer_GetUVSetChannels(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetUVSetChannels(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
             ::FbxLayerElementTypeArray*sipRes;

            sipRes = new  ::FbxLayerElementTypeArray(sipCpp->GetUVSetChannels());

            return sipConvertFromNewType(sipRes,sipType_FbxLayerElementTypeArray,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetUVSetChannels, doc_FbxLayer_GetUVSetChannels);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetUVSets, "GetUVSets(self) -> List");

extern "C" {static PyObject *meth_FbxLayer_GetUVSets(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetUVSets(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
            PyObject * sipRes = 0;

#line 15 "/home/ke/Documents/expressive-humanoid/sip/fbxlayer.sip"
        FbxArray<FbxLayerElementUV const*> lUVSets = sipCpp->GetUVSets();
        if ((sipRes = PyList_New(lUVSets.GetCount())) == NULL)
            return NULL;
        
        for (int i = 0; i < lUVSets.GetCount(); ++i)
        {
            FbxLayerElementUV * lTemp = new FbxLayerElementUV(*(lUVSets[i]));
            PyList_SET_ITEM(sipRes, i, sipConvertFromNewType(lTemp, sipType_FbxLayerElementUV, NULL));
        }
#line 294 "/home/ke/Documents/expressive-humanoid/build/Python38_x64/sipfbxFbxLayer.cpp"

            return sipRes;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetUVSets, doc_FbxLayer_GetUVSets);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetVertexColors, "GetVertexColors(self) -> FbxLayerElementVertexColor");

extern "C" {static PyObject *meth_FbxLayer_GetVertexColors(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetVertexColors(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
             ::FbxLayerElementVertexColor*sipRes;

            sipRes = sipCpp->GetVertexColors();

            return sipConvertFromType(sipRes,sipType_FbxLayerElementVertexColor,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetVertexColors, doc_FbxLayer_GetVertexColors);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetSmoothing, "GetSmoothing(self) -> FbxLayerElementSmoothing");

extern "C" {static PyObject *meth_FbxLayer_GetSmoothing(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetSmoothing(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
             ::FbxLayerElementSmoothing*sipRes;

            sipRes = sipCpp->GetSmoothing();

            return sipConvertFromType(sipRes,sipType_FbxLayerElementSmoothing,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetSmoothing, doc_FbxLayer_GetSmoothing);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetVertexCrease, "GetVertexCrease(self) -> FbxLayerElementCrease");

extern "C" {static PyObject *meth_FbxLayer_GetVertexCrease(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetVertexCrease(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
             ::FbxLayerElementCrease*sipRes;

            sipRes = sipCpp->GetVertexCrease();

            return sipConvertFromType(sipRes,sipType_FbxLayerElementCrease,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetVertexCrease, doc_FbxLayer_GetVertexCrease);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetEdgeCrease, "GetEdgeCrease(self) -> FbxLayerElementCrease");

extern "C" {static PyObject *meth_FbxLayer_GetEdgeCrease(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetEdgeCrease(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
             ::FbxLayerElementCrease*sipRes;

            sipRes = sipCpp->GetEdgeCrease();

            return sipConvertFromType(sipRes,sipType_FbxLayerElementCrease,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetEdgeCrease, doc_FbxLayer_GetEdgeCrease);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetVisibility, "GetVisibility(self) -> FbxLayerElementVisibility");

extern "C" {static PyObject *meth_FbxLayer_GetVisibility(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetVisibility(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxLayer, &sipCpp))
        {
             ::FbxLayerElementVisibility*sipRes;

            sipRes = sipCpp->GetVisibility();

            return sipConvertFromType(sipRes,sipType_FbxLayerElementVisibility,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetVisibility, doc_FbxLayer_GetVisibility);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetTextures, "GetTextures(self, FbxLayerElement.EType) -> FbxLayerElementTexture");

extern "C" {static PyObject *meth_FbxLayer_GetTextures(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetTextures(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElement::EType a0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BE", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElement_EType, &a0))
        {
             ::FbxLayerElementTexture*sipRes;

            sipRes = sipCpp->GetTextures(a0);

            return sipConvertFromType(sipRes,sipType_FbxLayerElementTexture,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetTextures, doc_FbxLayer_GetTextures);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_SetTextures, "SetTextures(self, FbxLayerElement.EType, FbxLayerElementTexture)");

extern "C" {static PyObject *meth_FbxLayer_SetTextures(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_SetTextures(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElement::EType a0;
         ::FbxLayerElementTexture* a1;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BEJ8", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElement_EType, &a0, sipType_FbxLayerElementTexture, &a1))
        {
            sipCpp->SetTextures(a0,a1);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_SetTextures, doc_FbxLayer_SetTextures);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_GetLayerElementOfType, "GetLayerElementOfType(self, FbxLayerElement.EType, bool = False) -> FbxLayerElement");

extern "C" {static PyObject *meth_FbxLayer_GetLayerElementOfType(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_GetLayerElementOfType(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElement::EType a0;
        bool a1 = 0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BE|b", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElement_EType, &a0, &a1))
        {
             ::FbxLayerElement*sipRes;

            sipRes = sipCpp->GetLayerElementOfType(a0,a1);

            return sipConvertFromType(sipRes,sipType_FbxLayerElement,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_GetLayerElementOfType, doc_FbxLayer_GetLayerElementOfType);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_SetNormals, "SetNormals(self, FbxLayerElementNormal)");

extern "C" {static PyObject *meth_FbxLayer_SetNormals(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_SetNormals(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementNormal* a0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElementNormal, &a0))
        {
            sipCpp->SetNormals(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_SetNormals, doc_FbxLayer_SetNormals);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_SetBinormals, "SetBinormals(self, FbxLayerElementBinormal)");

extern "C" {static PyObject *meth_FbxLayer_SetBinormals(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_SetBinormals(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementBinormal* a0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElementBinormal, &a0))
        {
            sipCpp->SetBinormals(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_SetBinormals, doc_FbxLayer_SetBinormals);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_SetTangents, "SetTangents(self, FbxLayerElementTangent)");

extern "C" {static PyObject *meth_FbxLayer_SetTangents(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_SetTangents(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementTangent* a0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElementTangent, &a0))
        {
            sipCpp->SetTangents(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_SetTangents, doc_FbxLayer_SetTangents);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_SetMaterials, "SetMaterials(self, FbxLayerElementMaterial)");

extern "C" {static PyObject *meth_FbxLayer_SetMaterials(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_SetMaterials(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementMaterial* a0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElementMaterial, &a0))
        {
            sipCpp->SetMaterials(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_SetMaterials, doc_FbxLayer_SetMaterials);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_SetPolygonGroups, "SetPolygonGroups(self, FbxLayerElementPolygonGroup)");

extern "C" {static PyObject *meth_FbxLayer_SetPolygonGroups(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_SetPolygonGroups(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementPolygonGroup* a0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElementPolygonGroup, &a0))
        {
            sipCpp->SetPolygonGroups(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_SetPolygonGroups, doc_FbxLayer_SetPolygonGroups);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_SetUVs, "SetUVs(self, FbxLayerElementUV, FbxLayerElement.EType = FbxLayerElement.eTextureDiffuse)");

extern "C" {static PyObject *meth_FbxLayer_SetUVs(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_SetUVs(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementUV* a0;
         ::FbxLayerElement::EType a1 = FbxLayerElement::eTextureDiffuse;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8|E", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElementUV, &a0, sipType_FbxLayerElement_EType, &a1))
        {
            sipCpp->SetUVs(a0,a1);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_SetUVs, doc_FbxLayer_SetUVs);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_SetVertexColors, "SetVertexColors(self, FbxLayerElementVertexColor)");

extern "C" {static PyObject *meth_FbxLayer_SetVertexColors(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_SetVertexColors(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementVertexColor* a0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElementVertexColor, &a0))
        {
            sipCpp->SetVertexColors(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_SetVertexColors, doc_FbxLayer_SetVertexColors);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_SetSmoothing, "SetSmoothing(self, FbxLayerElementSmoothing)");

extern "C" {static PyObject *meth_FbxLayer_SetSmoothing(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_SetSmoothing(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementSmoothing* a0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElementSmoothing, &a0))
        {
            sipCpp->SetSmoothing(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_SetSmoothing, doc_FbxLayer_SetSmoothing);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_SetVertexCrease, "SetVertexCrease(self, FbxLayerElementCrease)");

extern "C" {static PyObject *meth_FbxLayer_SetVertexCrease(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_SetVertexCrease(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementCrease* a0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElementCrease, &a0))
        {
            sipCpp->SetVertexCrease(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_SetVertexCrease, doc_FbxLayer_SetVertexCrease);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_SetEdgeCrease, "SetEdgeCrease(self, FbxLayerElementCrease)");

extern "C" {static PyObject *meth_FbxLayer_SetEdgeCrease(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_SetEdgeCrease(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementCrease* a0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElementCrease, &a0))
        {
            sipCpp->SetEdgeCrease(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_SetEdgeCrease, doc_FbxLayer_SetEdgeCrease);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_SetVisibility, "SetVisibility(self, FbxLayerElementVisibility)");

extern "C" {static PyObject *meth_FbxLayer_SetVisibility(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_SetVisibility(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElementVisibility* a0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElementVisibility, &a0))
        {
            sipCpp->SetVisibility(a0);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_SetVisibility, doc_FbxLayer_SetVisibility);

    return NULL;
}


PyDoc_STRVAR(doc_FbxLayer_CreateLayerElementOfType, "CreateLayerElementOfType(self, FbxLayerElement.EType, bool = False) -> FbxLayerElement");

extern "C" {static PyObject *meth_FbxLayer_CreateLayerElementOfType(PyObject *, PyObject *);}
static PyObject *meth_FbxLayer_CreateLayerElementOfType(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxLayerElement::EType a0;
        bool a1 = 0;
         ::FbxLayer *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BE|b", &sipSelf, sipType_FbxLayer, &sipCpp, sipType_FbxLayerElement_EType, &a0, &a1))
        {
             ::FbxLayerElement*sipRes;

            sipRes = sipCpp->CreateLayerElementOfType(a0,a1);

            return sipConvertFromType(sipRes,sipType_FbxLayerElement,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxLayer, sipName_CreateLayerElementOfType, doc_FbxLayer_CreateLayerElementOfType);

    return NULL;
}


/* Call the instance's destructor. */
extern "C" {static void release_FbxLayer(void *, int);}
static void release_FbxLayer(void *, int)
{
}


static PyMethodDef methods_FbxLayer[] = {
    {SIP_MLNAME_CAST(sipName_CreateLayerElementOfType), meth_FbxLayer_CreateLayerElementOfType, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_CreateLayerElementOfType)},
    {SIP_MLNAME_CAST(sipName_GetBinormals), meth_FbxLayer_GetBinormals, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetBinormals)},
    {SIP_MLNAME_CAST(sipName_GetEdgeCrease), meth_FbxLayer_GetEdgeCrease, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetEdgeCrease)},
    {SIP_MLNAME_CAST(sipName_GetLayerElementOfType), meth_FbxLayer_GetLayerElementOfType, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetLayerElementOfType)},
    {SIP_MLNAME_CAST(sipName_GetMaterials), meth_FbxLayer_GetMaterials, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetMaterials)},
    {SIP_MLNAME_CAST(sipName_GetNormals), meth_FbxLayer_GetNormals, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetNormals)},
    {SIP_MLNAME_CAST(sipName_GetPolygonGroups), meth_FbxLayer_GetPolygonGroups, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetPolygonGroups)},
    {SIP_MLNAME_CAST(sipName_GetSmoothing), meth_FbxLayer_GetSmoothing, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetSmoothing)},
    {SIP_MLNAME_CAST(sipName_GetTangents), meth_FbxLayer_GetTangents, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetTangents)},
    {SIP_MLNAME_CAST(sipName_GetTextures), meth_FbxLayer_GetTextures, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetTextures)},
    {SIP_MLNAME_CAST(sipName_GetUVSetChannels), meth_FbxLayer_GetUVSetChannels, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetUVSetChannels)},
    {SIP_MLNAME_CAST(sipName_GetUVSetCount), meth_FbxLayer_GetUVSetCount, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetUVSetCount)},
    {SIP_MLNAME_CAST(sipName_GetUVSets), meth_FbxLayer_GetUVSets, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetUVSets)},
    {SIP_MLNAME_CAST(sipName_GetUVs), meth_FbxLayer_GetUVs, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetUVs)},
    {SIP_MLNAME_CAST(sipName_GetVertexColors), meth_FbxLayer_GetVertexColors, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetVertexColors)},
    {SIP_MLNAME_CAST(sipName_GetVertexCrease), meth_FbxLayer_GetVertexCrease, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetVertexCrease)},
    {SIP_MLNAME_CAST(sipName_GetVisibility), meth_FbxLayer_GetVisibility, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_GetVisibility)},
    {SIP_MLNAME_CAST(sipName_SetBinormals), meth_FbxLayer_SetBinormals, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_SetBinormals)},
    {SIP_MLNAME_CAST(sipName_SetEdgeCrease), meth_FbxLayer_SetEdgeCrease, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_SetEdgeCrease)},
    {SIP_MLNAME_CAST(sipName_SetMaterials), meth_FbxLayer_SetMaterials, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_SetMaterials)},
    {SIP_MLNAME_CAST(sipName_SetNormals), meth_FbxLayer_SetNormals, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_SetNormals)},
    {SIP_MLNAME_CAST(sipName_SetPolygonGroups), meth_FbxLayer_SetPolygonGroups, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_SetPolygonGroups)},
    {SIP_MLNAME_CAST(sipName_SetSmoothing), meth_FbxLayer_SetSmoothing, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_SetSmoothing)},
    {SIP_MLNAME_CAST(sipName_SetTangents), meth_FbxLayer_SetTangents, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_SetTangents)},
    {SIP_MLNAME_CAST(sipName_SetTextures), meth_FbxLayer_SetTextures, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_SetTextures)},
    {SIP_MLNAME_CAST(sipName_SetUVs), meth_FbxLayer_SetUVs, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_SetUVs)},
    {SIP_MLNAME_CAST(sipName_SetVertexColors), meth_FbxLayer_SetVertexColors, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_SetVertexColors)},
    {SIP_MLNAME_CAST(sipName_SetVertexCrease), meth_FbxLayer_SetVertexCrease, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_SetVertexCrease)},
    {SIP_MLNAME_CAST(sipName_SetVisibility), meth_FbxLayer_SetVisibility, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxLayer_SetVisibility)}
};


sipClassTypeDef sipTypeDef_fbx_FbxLayer = {
    {
        -1,
        0,
        0,
        SIP_TYPE_CLASS,
        sipNameNr_FbxLayer,
        {0},
        0
    },
    {
        sipNameNr_FbxLayer,
        {0, 0, 1},
        29, methods_FbxLayer,
        0, 0,
        0, 0,
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
    0,
    -1,
    -1,
    0,
    0,
    0,
    0,
    0,
#if PY_MAJOR_VERSION >= 3
    0,
    0,
#else
    0,
    0,
    0,
    0,
#endif
    0,
    0,
    0,
    0,
    release_FbxLayer,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};
